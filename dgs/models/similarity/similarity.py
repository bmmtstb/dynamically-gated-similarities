"""Base class for Modules that compute any similarity."""

from abc import abstractmethod

import torch
from torch import nn

from dgs.models.module import BaseModule
from dgs.utils.states import DataSample
from dgs.utils.types import Config, NodePath, Validations

similarity_validations: Validations = {
    "module_name": [str],
}


class SimilarityModule(BaseModule, nn.Module):
    """Abstract class for similarity functions."""

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config, path)
        nn.Module.__init__(self)

        self.validate_params(similarity_validations)

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def get_data(self, ds: DataSample) -> any:
        """Get the data used in this similarity module."""
        raise NotImplementedError

    @abstractmethod
    def get_target(self, ds: DataSample) -> any:
        """Get the data used in this similarity module."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: DataSample, target: DataSample) -> torch.Tensor:
        """Compute the similarity between two input tensors."""
        raise NotImplementedError
