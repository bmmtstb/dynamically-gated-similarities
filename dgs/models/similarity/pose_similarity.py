"""
Modules for computing the similarity between two poses.
"""

import torch as t
from torchvision.ops import box_area, box_iou
from torchvision.transforms.v2 import ConvertBoundingBoxFormat
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from dgs.models.similarity.similarity import SimilarityModule
from dgs.utils.config import DEF_VAL
from dgs.utils.constants import OKS_SIGMAS
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

oks_validations: Validations = {
    "format": [str, ("in", list(OKS_SIGMAS.keys()))],
    # optional
    "keypoint_dim": ["optional", int, ("within", (1, 3))],
}

iou_validations: Validations = {}


class ObjectKeypointSimilarity(SimilarityModule):
    """Compute the object key-point similarity (OKS) between two batches of poses / States.

    Params
    ------

    format (str):
        The key point format, e.g., 'coco', 'coco-whole', ... has to be in OKS_SIGMAS.keys().

    Optional Params
    ---------------

    keypoint_dim (int, optional):
        The dimensionality of the key points. So whether 2D or 3D is expected.
        Default ``DEF_VAL.similarity.oks.kp_dim``.
    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(oks_validations)

        # get sigma
        sigma: t.Tensor = OKS_SIGMAS[self.params["format"]].to(device=self.device, dtype=self.precision)
        # With k = 2 * sigma -> shape [J]
        # We know that k is constant and k^2 is only ever required. Therefore, save it as parameter / buffer.
        self.register_buffer("k2", t.square(t.mul(2, sigma)))

        # Create a small value for epsilon to make sure that we do not divide by zero later on.
        self.register_buffer("eps", t.tensor(t.finfo(self.precision).eps, device=self.device, dtype=self.precision))
        # Set up a transform function to convert the bounding boxes if they have the wrong format
        self.transf_bbox_to_xyxy = ConvertBoundingBoxFormat("XYXY")

        self.kp_dim: int = self.params.get("keypoint_dim", DEF_VAL["similarity"]["oks"]["kp_dim"])

    def get_data(self, ds: State) -> t.Tensor:
        """Given a :class:`.State`, compute the detected / predicted key points with shape ``[B1 x J x 2|3]``
        and the areas of the respective ground-truth bounding-boxes with shape ``[B1]``.
        """
        return ds.keypoints.float().view(ds.B, -1, self.kp_dim)

    def get_area(self, ds: State) -> t.Tensor:
        """Given a :class:`.State`, compute the area of the bounding box."""
        bboxes = ds.bbox

        if bboxes.format == BoundingBoxFormat.XYXY:
            area = box_area(bboxes).float()  # (x2-x1) * (y2-y1)
        elif bboxes.format == BoundingBoxFormat.XYWH:
            area = bboxes[:, -2] * bboxes[:, -1]  # w * h
        else:
            bboxes = self.transf_bbox_to_xyxy(bboxes)
            area = box_area(bboxes).float()

        return area

    def get_target(self, ds: State) -> tuple[t.Tensor, t.Tensor]:
        """Given a :class:`.State` obtain the ground truth key points and the key-point-visibility.
        Both are tensors, the key points are a FloatTensor of shape ``[B2 x J x 2|3]``
        and the visibility is a BoolTensor of shape ``[B2 x J]``.
        """
        kps = ds.keypoints.float().view(ds.B, -1, self.kp_dim)
        vis = ds.cast_joint_weight(dtype=t.bool).squeeze(-1).view(ds.B, -1)
        return kps, vis

    def forward(self, data: State, target: State) -> t.Tensor:
        r"""Compute the object key-point similarity between a ground truth label and detected key points.

        There has to be one key point of the label for any detection. (Batch sizes have to match)

        Notes:
            Compute the key-point similarity :math:`\mathtt{ks}_i` for every joint between every detection and the
            respective ground truth annotation.

            .. math::
                \mathtt{ks}_i = \exp(-\dfrac{d_i^2}{2s^2k_i^2})

            The key-point similarity :math:`\mathtt{OKS}` is then computed as the weighted sum
            using the key-point visibilities as weights.

            .. math::
                \mathtt{OKS} = \dfrac{\sum_i \mathtt{ks}_i \cdot \delta (v_i > 0)}{\sum_i \delta (v_i > 0)}

            * :math:`d_i` the euclidean distance between the ground truth and detected key point
            * :math:`k_i` the constant for the key point, computed as :math:`k=2\cdot\sigma`
            * :math:`v_i` the visibility of the key point, with
                * 0 = unlabeled
                * 1 = labeled but not visible
                * 2 = labeled but visible
            * :math:`s` the scale of the ground truth object, with :math:`s^2` becoming the object's segmented area

        Args:
            data: A :class:`.State` object containing at least the key points and the bounding box. Shape ``N``.
            target: A :class:`.State` containing at least the target key points. Shape ``T``.

        Returns:
            A (Float)Tensor of shape ``[N x T]`` with values in ``[0..1]``.
            If requested, the softmax is computed along the -1 dimension,
            resulting in probability distributions for each value of the input data.
        """
        # get predicted key-points as [N x J x 2] and bbox area as [N]
        pred_kps = self.get_data(ds=data)
        bbox_area = self.get_area(ds=data)
        # get ground-truth key-points as [T x J x 2] and visibility as [T x J]
        gt_kps, gt_vis = self.get_target(ds=target)
        assert pred_kps.size(-1) == gt_kps.size(-1), "Key-points should have the same number of dimensions"
        # Compute d = Euclidean dist, but don't compute the sqrt, because only d^2 is required.
        # A little tensor magic, because if N != T and N != 1 and T != 1, regular subtraction will fail!
        # Therefore, modify the tensors to have shape [N x J x 2 x 1], [(1 x) J x 2 x T].
        # The output has shape [N x J x 2 x T], then square and sum over the number of dimensions (-2).
        d2 = t.sum(
            t.sub(pred_kps.unsqueeze(-1), gt_kps.permute(1, 2, 0)).square(),
            dim=-2,
        )  # -> [N x J x T]
        # Ground truth scale as bounding box area in relation to the image area it lies within.
        # Keep area s^2, because s is never used.
        s2 = bbox_area.flatten()  # [N]
        # Keypoint similarity for every key-point pair of ground truth and detected.
        # Use outer product to combine s^2 [N] with k^2 [J] and add epsilon to make sure to have non-zero values.
        # Again, modify the tensor shapes to match for division.
        # Shapes: d2 [N x J x T], new_outer [N x J x 1]
        ks = t.exp(-t.div(d2, (2 * t.outer(s2, self.k2) + self.eps).unsqueeze(-1)))  # -> [N x J x T]
        # The count of non-zero visibilities in the ground-truth
        count = t.count_nonzero(gt_vis, dim=-1)  # [T]
        # with ks [N x J x T], sum over all J and divide by the nof visibilities
        return self.softmax(t.div(t.where(gt_vis.T, ks, 0).sum(dim=-2), count).nan_to_num_(nan=0.0, posinf=0.0))


class IntersectionOverUnion(SimilarityModule):
    """Use the bounding-box based intersection-over-union as a similarity metric.

    Params
    ------

    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.bbox_transform = ConvertBoundingBoxFormat("XYXY")

    def get_data(self, ds: State) -> BoundingBoxes:
        """Given a :class:`.State` obtain the ground-truth bounding-boxes as
        :class:`torchvision.tv_tensors.BoundingBoxes` object of size ``[N x 4]``.

        Notes:
            The box_iou function expects that the bounding boxes are in the 'XYXY' format.
        """
        bboxes = ds.bbox
        if bboxes.format != BoundingBoxFormat.XYXY:
            bboxes = self.bbox_transform(bboxes)
        return bboxes

    def get_target(self, ds: State) -> BoundingBoxes:
        """Given a :class:`.State` obtain the ground-truth bounding-boxes as
        :class:`torchvision.tv_tensors.BoundingBoxes` object of size ``[T x 4]``.

        Notes:
            The function :func:`box_iou` expects that the bounding boxes are in the 'XYXY' format.
        """
        bboxes = ds.bbox
        if bboxes.format != BoundingBoxFormat.XYXY:
            bboxes = self.bbox_transform(bboxes)
        return bboxes

    def forward(self, data: State, target: State) -> t.Tensor:
        """Given two states containing bounding-boxes, compute the intersection over union between each pair.

        Args:
            data: A :class:`.State` object containing the detected bounding-boxes. Size ``N``
            target: A :class:`.State` object containing the target bounding-boxes. Size ``T``

        Returns:
            A (Float)Tensor of shape ``[N x T]`` with values in ``[0..1]``.
            If requested, the softmax is computed along the -1 dimension,
            resulting in probability distributions for each value of the input data.
        """
        return self.softmax(box_iou(self.get_data(ds=data), self.get_target(ds=target)))
