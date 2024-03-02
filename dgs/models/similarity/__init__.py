"""
Modules for handling similarity functions or other models that return similarity scores between two (or more) inputs.
"""

from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .pose_similarity import IntersectionOverUnion, ObjectKeypointSimilarity
from .similarity import SimilarityModule
from .torchreid import TorchreidSimilarity


def get_similarity_module(name: str) -> Type[SimilarityModule]:
    """Given the name of one of the SimilarityModules, return an instance."""
    modules: dict[str, Type[SimilarityModule]] = {
        "torchreid": TorchreidSimilarity,
        "iou": IntersectionOverUnion,
        "IntersectionOverUnion": IntersectionOverUnion,
        "oks": ObjectKeypointSimilarity,
        "ObjectKeypointSimilarity": ObjectKeypointSimilarity,
    }
    if name not in modules:
        raise InvalidParameterException(f"Unknown similarity with name: {name}.")

    return modules[name]
