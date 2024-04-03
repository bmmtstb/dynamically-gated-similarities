"""
Base module for different backbone models.

Most backbone models will predict values for:
    - bbox coordinates and using those, extract the image crop(s)
    - joint coordinates

Some models will additionally predict values for:
    - joint-heatmaps
    - joint confidence scores
    - joint visibility scores

Every Model should have capabilities to choose between different operation-modes, which can be set through the config:
    - compute the required values on the fly
    - precompute and save the values, so the saved data can later be loaded, possibly speeding up the training process
"""

from abc import abstractmethod

from dgs.models.module import BaseModule
from dgs.utils.state import State


class BaseBackboneModule(BaseModule):
    """Abstract class for backbone models"""

    def __call__(self, *args, **kwargs) -> State:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, state: State, *args, **kwargs) -> State:
        """Obtain the model outputs for the current iteration.

        Args:
            state: A State containing at least the path or paths pointing to one or multiple images.

        Returns:
            A :class:`.State` obtained through this backbone module, containing more data than before.
            Some backbone modules might compute bounding boxes and or key-point coordinates.
            Others might also (pre-) compute other data, which will be used later.
        """
        raise NotImplementedError
