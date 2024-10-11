"""
Modules for predicting the values of the alpha gates given :class:`State` s as input.
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from dgs.utils.types import Instance
from .alpha import BaseAlphaModule
from .combined import SequentialCombinedAlpha
from .fully_connected import FullyConnectedAlpha

__all__ = ["ALPHA_MODULES", "get_alpha_module", "register_alpha_module"]


ALPHA_MODULES: dict[str, Type[BaseAlphaModule]] = {
    "FullyConnectedAlpha": FullyConnectedAlpha,
    "FCA": FullyConnectedAlpha,  # shorthand alias
    "SequentialCombinedAlpha": SequentialCombinedAlpha,
    "SCA": SequentialCombinedAlpha,  # shorthand alias
}


def get_alpha_module(name: Instance) -> Type[BaseAlphaModule]:
    """Given the name of one alpha module, return an instance."""
    return get_instance(instance=name, instances=ALPHA_MODULES, inst_class=BaseAlphaModule)


def register_alpha_module(name: str, new_combine: Type[BaseAlphaModule]) -> None:
    """Register a new alpha module in :data:``ALPHA_MODULES``, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_combine, instances=ALPHA_MODULES, inst_class=BaseAlphaModule)
