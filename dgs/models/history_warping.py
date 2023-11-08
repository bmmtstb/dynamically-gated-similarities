"""
helpers and models for warping the pose-state of a track into the next time frame
"""
from abc import abstractmethod

import torch

from dgs.models.model import BaseModule
from dgs.models.pose_state import PoseState
from dgs.utils.types import Config


class HistoryWarpingModel(BaseModule):
    """
    Base class for other history warping classes
    """

    def __init__(self, config: Config, path: list[str]):
        super().__init__(config, path)

    def __call__(self, *args, **kwargs) -> PoseState:
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, pose: torch.Tensor, jcs: torch.Tensor, bbox: torch.Tensor) -> PoseState:
        """

        Parameters
        ----------
        pose: tensor of shape [EP x J x 2]
            History of poses per track

        jcs: tensor of shape [EP x J x 1]
            History of JCS per track

        bbox: tensor of shape [EP x 4]
            History of bboxes per track


        Returns
        -------
        next pose state
        """

        raise NotImplementedError


class HWKalmanFilter(HistoryWarpingModel):
    """
    Kalman Filter for simple History Warping
    """

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, pose: torch.Tensor, jcs: torch.Tensor, bbox: torch.Tensor) -> PoseState:
        raise NotImplementedError


def get_warping_model(config: Config, *args, **kwargs) -> HistoryWarpingModel:
    """
    Given config, set up the current HistoryWarping model

    Returns
    -------
    The chosen model on the device given by the config
    """
    if config.warp_model == "kalman":
        return HWKalmanFilter(config=config, *args, **kwargs)
    raise NotImplementedError
