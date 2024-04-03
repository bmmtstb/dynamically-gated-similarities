"""
Base module for different backbone models.

All backbone models have to predict values for:
    - bbox coordinates

Most backbone models will predict values for:
    - joint coordinates
    - using the bbox coordinates to extract the image crop(s)

Some models will additionally predict values for:
    - joint-heatmaps
    - joint confidence scores
    - joint visibility scores
"""

from abc import abstractmethod

from dgs.models.module import BaseModule
from dgs.utils.state import State
from dgs.utils.types import FilePaths


class BaseBackboneModule(BaseModule):
    """Abstract class for backbone models"""

    def __call__(self, *args, **kwargs) -> State:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, image_paths: FilePaths, *args, **kwargs) -> State:
        """Obtain the model outputs for the current iteration.

        Args:
            image_paths: A tuple containing the path or paths pointing to one or multiple images.

        Returns:
            A :class:`.State` obtained through this backbone module, containing more data than before.
            Some backbone modules might compute bounding boxes and or key-point coordinates.
            Others might also (pre-) compute other data, which will be used later.
        """
        raise NotImplementedError
