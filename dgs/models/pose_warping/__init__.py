"""
Module to warp a given pose, pose-state -
or more generally to predict the next pose of a person given previous time steps.
"""
from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .kalman import KalmanFilterWarpingModel
from .pose_warping import PoseWarpingModule


def get_pose_warping(name: str) -> Type[PoseWarpingModule]:
    """Given the name of one pose-warping module, return an instance."""
    if name == "Kalman":
        return KalmanFilterWarpingModel
    raise InvalidParameterException(f"Unknown pose warping module with name: {name}.")
