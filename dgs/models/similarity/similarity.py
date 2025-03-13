"""Base class for Modules that compute any similarity."""

from abc import abstractmethod

import torch as t
from torch import nn

from dgs.models.module import enable_keyboard_interrupt
from dgs.models.modules.named import NamedModule
from dgs.utils.config import DEF_VAL
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

similarity_validations: Validations = {
    # optional
    "softmax": ["optional", bool],
    "train_key": ["optional", str],
}


class SimilarityModule(NamedModule, nn.Module):
    """Abstract class for similarity functions.

    Params
    ------

    module_name (str):
        The name of the similarity module.

    Optional Params
    ---------------

    softmax (bool, optional):
        Whether to apply the softmax function to the (batched) output of the similarity function.
        Default ``DEF_VAL.similarity.softmax``.
    train_key (str, optional):
        A name of a :class:`.State` property to use to retrieve the data during training.
        E.g. usage of :meth:`State.bbox_relative` instead of the regular bbox.
        If this value isn't set, the regular :meth:`SimilarityModule.get_data` call is used.

    """

    softmax: nn.Sequential

    def __init__(self, config: Config, path: NodePath):
        NamedModule.__init__(self, config, path)
        nn.Module.__init__(self)

        self.validate_params(similarity_validations)

        softmax = nn.Sequential()
        if self.params.get("softmax", DEF_VAL["similarity"]["softmax"]):
            softmax.append(nn.Softmax(dim=-1))
        self.register_module(name="softmax", module=self.configure_torch_module(softmax))

    @property
    def module_type(self) -> str:
        return "similarity"

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

    def get_train_data(self, ds: State) -> any:
        """A custom function to get special data for training purposes.
        If "train_key" is not given, uses the regular :func:`get_data` function of this module.
        """
        if "train_key" in self.params:
            return getattr(ds, self.params["train_key"])
        return self.get_data(ds)

    @abstractmethod
    @enable_keyboard_interrupt
    def forward(self, data: State, target: State) -> t.Tensor:
        """Compute the similarity between two input tensors. Make sure to compute the softmax if ``softmax`` is True."""
        raise NotImplementedError
