"""
Modules for running, training and testing multiple other modules.
"""

from typing import Type

from .engine import EngineModule
from .visual_sim_engine import VisualSimilarityEngine
from ...utils.exceptions import InvalidParameterException

ENGINES: dict[str, Type[EngineModule]] = {
    "VisualSimilarityEngine": VisualSimilarityEngine,
}


def get_engine(name: str) -> Type[EngineModule]:
    """Given the name of an engine module, return the type."""
    if name not in ENGINES:
        raise InvalidParameterException(f"Name '{name}' is not a valid engine module.")
    return ENGINES[name]
