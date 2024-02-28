"""
Modules for computing the similarity between two poses.
"""

import torch
from torchvision.ops import box_iou
from torchvision.transforms.v2 import ConvertBoundingBoxFormat
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from dgs.models.similarity.similarity import SimilarityModule
from dgs.utils.constants import OKS_SIGMAS
from dgs.utils.types import Config, NodePath, Validations

oks_validations: Validations = {"format": [str, ("in", list(OKS_SIGMAS.keys()))]}


class ObjectKeypointSimilarity(SimilarityModule):
    """Compute the object key-point similarity (OKS) between two poses.

    Params
    ------

    format (str):
        The key point format, e.g., 'coco', 'coco-whole', ... has to be in OKS_SIGMAS.keys().

    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(oks_validations)

        sigma: torch.Tensor = OKS_SIGMAS[self.params["format"]].to(device=self.device, dtype=torch.float32)
        self.register_buffer("sigma_const", sigma)

    def forward(self, data: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
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
            data: tuple containing the detected / predicted key points with shape ``[B1 x J x 2]``
                and the areas of the respective ground-truth bounding-boxes with shape ``[B1]``.
                To compute the bbox area, it is possible to use :class:`~torchvision.ops.box_area`.
            target: ground truth key points including visibility as third dimension with shape ``[B2 x J x 2]``
        """

        def extract_data(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """given key points with possible visibility. Key points can be batched.
            Possible input shape: [(B) x J x (2|3)]
            Return shape [B x J x 2] and [B x 1 x 1]
            """
            if d.ndim == 2:
                d = d.unsqueeze(0)

            if d.size(-1) == 3:
                return d.split([2, 1], dim=-1)
            return d, torch.ones((d.size(0), 1, 1), device=target.device)

        # get pred and gt key-points as [B x J x 2] and visibility as [B x 1 x 1]
        pred_kps, bbox_area = data
        pred_kps, _ = extract_data(pred_kps)
        gt_kps, gt_vis = extract_data(target)

        # d = euclidean dist, don't compute sqrt, because only d^2 is required.
        d2 = torch.sum(torch.pow(torch.subtract(gt_kps, pred_kps), 2), dim=-1)  # [B x J]
        # k = 2 * sig, only k^2 is required.
        k2 = torch.square(torch.mul(2, self.sigma_const))  # [J]
        # Ground truth scale as bounding box area in relation to the image area it lies within.
        # Keep area s^2, because s is never used.
        s2 = bbox_area  # [B]
        # Keypoint similarity for every key-point pair of ground truth and detected.
        # Use outer product to combine s^2 [B] with k^2 [J].
        # Add epsilon to make sure to have non-zero values.
        ks = torch.exp(-torch.div(d2, torch.mul(2, torch.outer(s2, k2)) + torch.finfo(torch.float32).eps))  # [B x J]
        # for every pair in B, sum over all J
        return torch.div(torch.sum(ks * torch.where(gt_vis > 0, 1.0, 0.0), dim=-1), torch.count_nonzero(gt_vis))


class IntersectionOverUnion(SimilarityModule):
    """Use the bounding-box based intersection-over-union as a similarity metric."""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.transform = ConvertBoundingBoxFormat("XYXY")

    def forward(self, data: BoundingBoxes, target: BoundingBoxes) -> torch.Tensor:
        if data.format != BoundingBoxFormat.XYXY:
            data = self.transform(data)
        if target.format != BoundingBoxFormat.XYXY:
            target = self.transform(target)
        return box_iou(data, target)
