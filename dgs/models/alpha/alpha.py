"""
Base class for modules that predict alpha values given a :class:`.State`.
"""

from abc import abstractmethod

import torch as t

from dgs.models.modules.named import NamedModule
from dgs.utils.state import State
from dgs.utils.torchtools import init_model_params, load_pretrained_weights
from dgs.utils.types import Config, NodePath, Validations

alpha_validations: Validations = {
    # optional
    "weight": ["optional", ("file exists", "./weights/")],
}


class BaseAlphaModule(NamedModule, t.nn.Module):
    """Given a state as input, compute and return the weight of the alpha gate.

    Optional Params
    ---------------

    weight (FilePath):
        Local or absolute path to the pretrained weights of the model.
        Can be left empty.

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
        """The call function uses :func:`sub_forward` and not :func:`forward`
        This way, the sequential layers can just be called later on.
        """
        return self.sub_forward(*args, **kwargs)

    @abstractmethod
    def forward(self, s: State) -> t.Tensor:
        raise NotImplementedError

    def sub_forward(self, data: t.Tensor) -> t.Tensor:
        """Function to call when module is called from within a combined alpha module."""
        if not hasattr(self, "model") or self.model is None:
            return data
        return self.model(data)

    @abstractmethod
    def get_data(self, s: State) -> any:
        """Given a state, return the data which is input into the model."""
        raise NotImplementedError

    def load_weights(self) -> None:
        """Load the weights of the model from the given file path. If no weights are given, initialize the model."""
        if "weight" in self.params:
            fp = self.params.get("weight")
            load_pretrained_weights(model=self.model, weight_path=fp)
        else:
            init_model_params(self.model)
