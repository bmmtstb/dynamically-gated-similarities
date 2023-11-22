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
from dgs.models.states import BackboneOutput


class BackboneModule(BaseModule):
    """Abstract class for backbone models"""

    def __call__(self, *args, **kwargs) -> ...:
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> BackboneOutput:
        """
        fixme: define the return type or class for backbone objects, because there will be plenty return values

        Obtain the model outputs for the current iteration.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def precompute_values(self) -> None:
        """
        Use the backbone model to precompute all the necessary values

        Predicts values for all given data and saves them to local files
        """
        raise NotImplementedError

    @abstractmethod
    def load_precomputed(self) -> ...:
        """Use the backbone model to load precomputed values exported by itself"""
        raise NotImplementedError
