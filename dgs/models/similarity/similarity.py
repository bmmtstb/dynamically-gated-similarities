"""Base class for Modules that compute any similarity."""

from abc import abstractmethod

import torch as t
from torch import nn

from dgs.models.module import BaseModule
from dgs.utils.config import DEF_VAL
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

similarity_validations: Validations = {
    "module_name": [str],
    # optional
    "softmax": ["optional", bool],
}


class SimilarityModule(BaseModule, nn.Module):
    """Abstract class for similarity functions.

    Params
    ------

    softmax (bool, optional):
        Whether to apply the softmax function to the (batched) output of the similarity function.
        Default ``DEF_VAL.similarity.softmax``.
    """

    softmax: nn.Sequential

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config, path)
        nn.Module.__init__(self)

        self.validate_params(similarity_validations)

        softmax = nn.Sequential()
        if self.params.get("softmax", DEF_VAL["similarity"]["softmax"]):
            softmax.append(nn.Softmax(dim=-1))
        self.register_module(name="softmax", module=self.configure_torch_module(softmax))

    def __call__(self, *args, **kwargs) -> t.Tensor:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def get_data(self, ds: State) -> any:
        """Get the data used in this similarity module."""
        raise NotImplementedError

    @abstractmethod
    def get_target(self, ds: State) -> any:
        """Get the data used in this similarity module."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: State, target: State) -> t.Tensor:
        """Compute the similarity between two input tensors. Make sure to compute the softmax if ``softmax`` is True."""
        raise NotImplementedError
