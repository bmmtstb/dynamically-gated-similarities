"""
Abstraction for different similarity functions.

Similarity functions compute a similarity score "likeness" between two equally sized inputs.
"""
from typing import Callable

import torch
from torch import nn
from torch.nn import CosineSimilarity, PairwiseDistance

from dgs.models.module import BaseModule
from dgs.utils.types import Config, NodePath


class SimilarityModule(BaseModule, nn.Module):
    """Abstract class for similarity functions.

    TODO sizes / shapes? input and output?
    """

    func: Callable[..., torch.Tensor]

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config, path)
        nn.Module.__init__(self)

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """Compute the similarity between two tensors using this module's specified function.

        Args:
            d1: The first tensor with data.
            d2: The second tensor with data.

        Returns:
            The similarity between `d1` and `d2`, as `torch.Tensor`.
        """
        return self.func(d1, d2)

    def get_kwargs(self) -> dict[str, any]:
        """If there are additional kwargs in `self.params`, return them."""
        kwargs: dict[str, any] = {}

        if "kwargs" in self.params and self.params["kwargs"] is not None:
            kwargs = self.params["kwargs"]
        return kwargs


class CosineSimilarityModule(SimilarityModule):
    __doc__ = CosineSimilarity.__doc__

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.func = CosineSimilarity(**self.get_kwargs())


class PairwiseDistanceModule(SimilarityModule):
    __doc__ = PairwiseDistance.__doc__

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.func = PairwiseDistance(**self.get_kwargs())


class PNormDistanceModule(SimilarityModule):
    __doc__ = torch.cdist.__doc__

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.kwargs = self.get_kwargs()

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """The cdist function accepts additional kwargs which might be changed by the user."""
        return torch.cdist(d1, d2, **self.kwargs)


class EuclideanDistanceModule(SimilarityModule):
    """Euclidean distance is :class:`torch.cdist` with :math:`p=2`."""

    __doc__ = torch.cdist.__doc__

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.kwargs = self.get_kwargs()
        self.kwargs["p"] = 2

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """The cdist function accepts additional kwargs which might be changed by the user."""
        return torch.cdist(d1, d2, **self.kwargs)


class DotProductModule(SimilarityModule):
    """Similarity using the dot product."""

    __doc__ = torch.tensordot.__doc__

    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """Use the function `torch.tensordot()` to compute the dot product of two tensors.

        Notes:
            With input shapes of d1 = ``[A x B x ... x C]`` and d2 = ``[D x E x ... x C]``
            this will result in the dot product having a shape of ``[A x B x ... x D x E x ...]``.
        """
        return torch.tensordot(d1, d2, dims=([-1], [-1]))
