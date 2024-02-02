"""
Implementation if kalman filter for basic pose warping
"""

import torch

from dgs.models.pose_warping.pose_warping import PoseWarpingModule
from dgs.models.states import PoseState


class KalmanFilterWarpingModel(PoseWarpingModule):
    """Kalman Filter for pose warping"""

    def forward(self, pose: torch.Tensor, jcs: torch.Tensor, bbox: torch.Tensor) -> PoseState:
        raise NotImplementedError
