"""
Modules for handling similarity functions or other models that return similarity scores between two (or more) inputs.
"""

from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .combined import CombinedSimilarityModule, DynamicallyGatedSimilarities, StaticAlphaWeightingModule
from .pose_similarity import ObjectKeypointSimilarity
from .similarity import SimilarityModule
from .torchreid import TorchreidSimilarity


def get_similarity_module(name: str) -> Type[SimilarityModule]:
    """Given the name of one of the SimilarityModules, return an instance."""
    modules: dict[str, Type[SimilarityModule]] = {
        "torchreid": TorchreidSimilarity,
    }
    if name not in modules:
        raise InvalidParameterException(f"Unknown similarity with name: {name}.")

    return modules[name]


def get_dgs_module(name: str) -> Type[CombinedSimilarityModule]:
    """Given the name of one module that combines different similarity modules, return an instance."""
    modules: dict[str, Type[CombinedSimilarityModule]] = {
        "DGS": DynamicallyGatedSimilarities,
        "dynamic_alpha": DynamicallyGatedSimilarities,  # synonym for DGS
        "static_alpha": StaticAlphaWeightingModule,
    }
    if name not in modules:
        raise InvalidParameterException(f"Unknown combined similarity module with name: {name}.")
    return modules[name]
