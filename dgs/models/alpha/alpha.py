"""
Base class for modules that predict alpha values given a :class:`State`.
"""

from abc import abstractmethod

import torch as t

from dgs.models.modules.named import NamedModule
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

alpha_validations: Validations = {}


class BaseAlphaModule(NamedModule, t.nn.Module):
    """Given a state as input, compute and return the weight of the alpha gate.

    Params
    ------

    Optional Params
    ---------------

    """

    model: t.nn.Module

    def __init__(self, config: Config, path: NodePath):
        NamedModule.__init__(self, config=config, path=path)
        t.nn.Module.__init__(self)

        self.validate_params(alpha_validations)

    @property
    def module_type(self) -> str:
        return "alpha"

    def __call__(self, *args, **kwargs) -> any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, s: State) -> t.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_data(self, s: State) -> any:
        """Given a state, return the data which is input into the model."""
        raise NotImplementedError
