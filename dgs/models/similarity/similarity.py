"""
Abstraction for different similarity functions.

Similarity functions compute a similarity score "likeness" between two equally sized inputs.
"""
from abc import abstractmethod

import torch

from dgs.models.module import BaseModule


class SimilarityModule(BaseModule):
    """Abstract class for similarity functions"""

    def __call__(self, *args, **kwargs) -> any:
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        TODO has to be extendable into multiple dimensions

        Args:
            *args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        """some similarity functions will not have weights"""
        raise NotImplementedError
