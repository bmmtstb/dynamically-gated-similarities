"""
Module to warp a given pose, pose-state - \
    or more generally to predict the next pose of a person given previous time steps.
"""

__all__ = ["KalmanFilterWarpingModel"]

from dgs.models.pose_warping.kalman import KalmanFilterWarpingModel
