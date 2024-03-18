"""
Use the DGS model to run multiple similarity modules (:py:class:`~.SimilarityModule`)
and combine them using the combine modules (:class:`~.CombineSimilaritiesModule`).
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from .dgs import DGSModule

DGS_MODULES: dict[str, Type[DGSModule]] = {
    "DGS": DGSModule,
}


def get_dgs_module(name: str) -> Type[DGSModule]:
    """Given the name of one DGS module, return an instance."""
    return get_instance(instance=name, instances=DGS_MODULES, inst_class=DGSModule)


def register_dgs_module(name: str, new_dgs: Type[DGSModule]) -> None:
    """Register a new DGS module module in :data:``DGS_MODULES``, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_dgs, instances=DGS_MODULES, inst_class=DGSModule)
