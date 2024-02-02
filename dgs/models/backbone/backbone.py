"""
Modules for different backbone models.

Every backbone model should predict values for:
    - imageCrop
    - joint-heatmaps
    - bbox shape
    - joint coordinates
    - (possibly) joint confidence score
    - (possibly) joint visibility score

Every Model should have capabilities to choose between different operation-modes, which can be set through the config:
    - precompute and save the values
    - load previously saved values
    - compute the values on the fly


Defaults to use AlphaPose as Backbone but should be extendable to use others.
"""

from abc import abstractmethod

from dgs.models.module import BaseModule
from dgs.models.states import DataSample


class BackboneModule(BaseModule):
    """Abstract class for backbone models"""

    def __call__(self, *args, **kwargs) -> DataSample:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, img_name: str, *args, **kwargs) -> DataSample:
        """Obtain the model outputs for the current iteration.

        Args:
            img_name: Name of the image to obtain results for.

        Returns:
            A `DataSample` obtained through this backbone module.
        """
        raise NotImplementedError
