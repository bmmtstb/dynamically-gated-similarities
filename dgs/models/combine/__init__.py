"""
Combine multiple similarity matrices.
Obtain similarity matrices as a result of one or multiple
:class:`~dgs.models.similarity.similarity.SimilarityModule`'s.
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from dgs.utils.types import Instance
from .combine import CombineSimilaritiesModule
from .dynamic import AlphaCombine, DynamicAlphaCombine
from .static import StaticAlphaCombine

__all__ = ["COMBINE_MODULES", "get_combine_module", "register_combine_module"]


COMBINE_MODULES: dict[str, Type[CombineSimilaritiesModule]] = {
    "alpha_combine": AlphaCombine,
    "dynamic_alpha": DynamicAlphaCombine,
    "dac": DynamicAlphaCombine,  # alias
    "static_alpha": StaticAlphaCombine,
    "constant_alpha": StaticAlphaCombine,  # alias
}


def get_combine_module(name: Instance) -> Type[CombineSimilaritiesModule]:
    """Given the name of one module that combines different similarity modules, return an instance."""
    return get_instance(instance=name, instances=COMBINE_MODULES, inst_class=CombineSimilaritiesModule)


def register_combine_module(name: str, new_combine: Type[CombineSimilaritiesModule]) -> None:
    """Register a new combine module in :data:``COMBINE_MODULES``, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_combine, instances=COMBINE_MODULES, inst_class=CombineSimilaritiesModule)
