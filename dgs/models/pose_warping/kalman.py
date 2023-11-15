"""
Implementation if kalman filter for basic pose warping
"""
import torch

from dgs.models.pose_state import PoseState
from dgs.models.pose_warping.pose_warping import PoseWarpingModule


class KalmanFilterWarpingModel(PoseWarpingModule):
    """Kalman Filter for pose warping"""

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, pose: torch.Tensor, jcs: torch.Tensor, bbox: torch.Tensor) -> PoseState:
        raise NotImplementedError
