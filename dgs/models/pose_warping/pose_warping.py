"""
Helpers and models for warping the pose-state of a track into the next time frame.
"""
from abc import abstractmethod

import torch

from dgs.models.module import BaseModule
from dgs.models.states import PoseState


class PoseWarpingModule(BaseModule):
    """
    Base class for pose warping modules

    The goal of pose warping is to predict the next PoseState given information about the last (few) states.
    """

    def __call__(self, *args, **kwargs) -> PoseState:
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, pose: torch.Tensor, jcs: torch.Tensor, bbox: torch.Tensor) -> PoseState:
        """

        Args:
            pose: History of poses per track as `torch.Tensor` of shape ``[EP x J x 2]``.
            jcs: History of JCS per track as `torch.Tensor` of shape ``[EP x J x 1]``.
            bbox: History of bboxes per track as `torch.Tensor` of shape ``[EP x 4]``.

        Returns:
            The next pose state.
        """

        raise NotImplementedError
