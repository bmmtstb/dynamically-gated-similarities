"""
Modules for computing the similarity between two poses.
"""

import torch
from torch import nn
from torchvision.ops import box_area, box_iou
from torchvision.transforms.v2 import ConvertBoundingBoxFormat
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from dgs.utils.constants import OKS_SIGMAS
from dgs.utils.states import DataSample
from dgs.utils.types import Config, NodePath, Validations
from .similarity import SimilarityModule

oks_validations: Validations = {"format": [str, ("in", list(OKS_SIGMAS.keys()))]}


class ObjectKeypointSimilarity(SimilarityModule):
    """Compute the object key-point similarity (OKS) between two batches of poses.

    Params
    ------

    format (str):
        The key point format, e.g., 'coco', 'coco-whole', ... has to be in OKS_SIGMAS.keys().

    softmax (bool, optional):
        Whether to compute the softmax of the result.
        All values will lie in the range :math:`[0, 1]`, with softmax, they additionally sum to one.
        Default False.
    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(oks_validations)

        # get sigma
        sigma: torch.Tensor = OKS_SIGMAS[self.params["format"]].to(device=self.device, dtype=torch.float32)
        # With k = 2 * sigma -> shape [J]
        # We know that k is constant and k^2 is only ever required. Therefore, save it as parameter / buffer.
        self.register_buffer("k2", torch.square(torch.mul(2, sigma)))

        # Create a small value for epsilon to make sure that we do not divide by zero later on.
        self.register_buffer(
            "eps", torch.tensor(torch.finfo(torch.float32).eps, device=self.device, dtype=torch.float32)
        )
        # Set up a transform function to convert the bounding boxes if they have the wrong format
        self.transform = ConvertBoundingBoxFormat("XYXY")

        # Set up softmax function if requested
        self.softmax = nn.Sequential()
        if self.params.get("softmax", False):
            self.softmax.append(nn.Softmax(dim=-1))

    def get_data(self, ds: DataSample) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a data sample, compute the detected / predicted key points with shape ``[B1 x J x 2]``
        and the areas of the respective ground-truth bounding-boxes with shape ``[B1]``.

        Notes:
            To compute the bbox area, it is possible to use :class:`~torchvision.ops.box_area`.
            For the box_area function, it is expected that the bounding boxes are given in 'XYXY' format.
        """
        kps = ds.keypoints.float().view(ds.B, -1, 2)

        bboxes = ds.bbox

        if bboxes.format == BoundingBoxFormat.XYXY:
            area = box_area(bboxes).float()  # (x2-x1) * (y2-y1)
        elif bboxes.format == BoundingBoxFormat.XYWH:
            area = bboxes[:, -2] * bboxes[:, -1]  # w * h
        else:
            bboxes = self.transform(bboxes)
            area = box_area(bboxes).float()

        return kps, area

    def get_target(self, ds: DataSample) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a DataSample obtain the ground truth key points and the key-point-visibility.
        Both are tensors, the key points are a FloatTensor of shape ``[B2 x J x 2]``
        and the visibility is a BoolTensor of shape ``[B2 x J]``.
        """
        kps = ds.keypoints.float().view(ds.B, -1, 2)
        vis = ds.cast_joint_weight(dtype=torch.bool).squeeze(-1).view(ds.B, -1)
        return kps, vis

    def forward(self, data: DataSample, target: DataSample) -> torch.Tensor:
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

        Fixme: exclude ignore regions from image_shape ?

        Args:
            data: A DataSample object containing at least the key points and the bounding box.
            target: A DataSample containing at least the target key points.
        """
        # get predicted key-points as [B1 x J x 2] and bbox area as [B1]
        pred_kps, bbox_area = self.get_data(ds=data)
        # get ground-truth key-points as [B2 x J x 2] and visibility as [B2 x J]
        gt_kps, gt_vis = self.get_target(ds=target)
        assert pred_kps.size(-1) == gt_kps.size(-1), "Key-points should have the same number of dimensions"
        # Compute d = Euclidean dist, but don't compute the sqrt, because only d^2 is required.
        # A little tensor magic, because if B1 != B2 and B1 != 1 and B2 != 1, regular subtraction will fail!
        # Therefore, modify the tensors to have shape [B1 x J x 2 x 1], [(1 x) J x 2 x B2].
        # The output has shape [B1 x J x 2 x B2], then square and sum over the number of dimensions (-2).
        d2 = torch.sum(
            torch.sub(pred_kps.unsqueeze(-1), gt_kps.permute(1, 2, 0)).square(),
            dim=-2,
        )  # -> [B1 x J x B2]
        # Ground truth scale as bounding box area in relation to the image area it lies within.
        # Keep area s^2, because s is never used.
        s2 = bbox_area.flatten()  # [B1]
        # Keypoint similarity for every key-point pair of ground truth and detected.
        # Use outer product to combine s^2 [B1] with k^2 [J] and add epsilon to make sure to have non-zero values.
        # Again, modify the tensor shapes to match for division.
        # Shapes: d2 [B1 x J x B2], new_outer [B1 x J x 1]
        ks = torch.exp(-torch.div(d2, (2 * torch.outer(s2, self.k2) + self.eps).unsqueeze(-1)))  # -> [B1 x J x B2]
        # The count of non-zero visibilities in the ground-truth
        count = torch.count_nonzero(gt_vis, dim=-1)  # [B2]
        # for every pair in B, sum over all J
        return self.softmax(torch.div(torch.where(gt_vis.T, ks, 0).sum(dim=-2), count).nan_to_num_(nan=0.0, posinf=0.0))


class IntersectionOverUnion(SimilarityModule):
    """Use the bounding-box based intersection-over-union as a similarity metric.

    Params
    ------

    softmax (bool, optional):
        Whether to compute the softmax of the result.
        All values will lie in the range :math:`[0, 1]`, with softmax, they additionally sum to one.
        Default False.
    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.transform = ConvertBoundingBoxFormat("XYXY")

        # Set up softmax function if requested
        self.softmax = nn.Sequential()
        if self.params.get("softmax", False):
            self.softmax.append(nn.Softmax(dim=-1))

    def get_data(self, ds: DataSample) -> BoundingBoxes:
        """Given a DataSample obtain the ground truth bounding boxes as BoundingBoxes object of size ``[B1 x 4]``.

        Notes:
            The box_iou function expects that the bounding boxes are in the 'XYXY' format.
        """
        bboxes = ds.bbox
        if bboxes.format != BoundingBoxFormat.XYXY:
            bboxes = self.transform(bboxes)
        return bboxes

    def get_target(self, ds: DataSample) -> BoundingBoxes:
        """Given a DataSample obtain the ground truth bounding boxes as BoundingBoxes object of size ``[B2 x 4]``.

        Notes:
            The box_iou function expects that the bounding boxes are in the 'XYXY' format.
        """
        bboxes = ds.bbox
        if bboxes.format != BoundingBoxFormat.XYXY:
            bboxes = self.transform(bboxes)
        return bboxes

    def forward(self, data: DataSample, target: DataSample) -> torch.Tensor:
        return self.softmax(box_iou(self.get_data(ds=data), self.get_target(ds=target)))
