"""
Implementation of modules that combine two or more similarity matrices.
Obtain similarity matrices as a result of one or multiple
:class:`~dgs.models.similarity.similarity.SimilarityModule`'s.
"""

from abc import abstractmethod

import torch as t
from torch import nn

from dgs.models.module import enable_keyboard_interrupt
from dgs.models.modules.named import NamedModule
from dgs.utils.config import DEF_VAL
from dgs.utils.types import Config, NodePath, Validations

combine_validations: Validations = {
    # optional
    "softmax": ["optional", bool],
}


class CombineSimilaritiesModule(NamedModule, nn.Module):
    """Given two or more similarity matrices, combine them into a single similarity matrix.

    Params
    ------

    Optional Params
    ---------------

    softmax (bool, optional):
        Whether to compute the softmax along the last dimension of the resulting weighted similarity matrix.
        Default ``DEF_VAL.combine.softmax``.

    """

    softmax: nn.Sequential

    def __init__(self, config: Config, path: NodePath):
        NamedModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(combine_validations)

        softmax = nn.Sequential()
        if self.params.get("softmax", DEF_VAL["combine"]["softmax"]):
            softmax.append(nn.Softmax(dim=-1))
        self.register_module(name="softmax", module=self.configure_torch_module(softmax))

    @property
    def module_type(self) -> str:
        return "combine"

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        return self.forward(*args, **kwargs)

    @abstractmethod
    @enable_keyboard_interrupt
    def forward(self, *args, **kwargs) -> t.Tensor:  # pragma: no cover
        raise NotImplementedError
