"""
Modules for running, training and testing multiple other modules.
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from .engine import EngineModule
from .visual_sim_engine import VisualSimilarityEngine
from ...utils.types import Instance

ENGINES: dict[str, Type[EngineModule]] = {
    "VisualSimilarityEngine": VisualSimilarityEngine,
}


def register_engine(name: str, new_engine: Type[EngineModule]) -> None:
    """Given a new engine and its name, register it in ``ENGINES`` to be able to use it from configuration files."""
    register_instance(name=name, instance=new_engine, instances=ENGINES, inst_class=EngineModule)


def get_engine(name: Instance) -> Type[EngineModule]:
    """Given the name of an engine module, return the type."""
    return get_instance(instance=name, instances=ENGINES, inst_class=EngineModule)
