"""
Helpers and models for warping the pose-state of a track into the next time frame.
"""
from abc import abstractmethod

from dgs.models.module import BaseModule
from dgs.models.states import PoseStates


class PoseWarpingModule(BaseModule):
    """
    Base class for pose warping modules

    The goal of pose warping is to predict the next PoseState given information about the last (few) states.
    """

    def __call__(self, *args, **kwargs) -> PoseStates:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, ps: PoseStates) -> PoseStates:
        """

        Args:
            ps: History of poses as `PoseState` object.

        Returns:
            The next pose state.
        """
        raise NotImplementedError
