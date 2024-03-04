"""
Modules for handling similarity functions or other models that return similarity scores between two (or more) inputs.
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from .pose_similarity import IntersectionOverUnion, ObjectKeypointSimilarity
from .similarity import SimilarityModule
from .torchreid import TorchreidSimilarity

SIMILARITIES: dict[str, Type[SimilarityModule]] = {
    "torchreid": TorchreidSimilarity,
    "iou": IntersectionOverUnion,
    "IntersectionOverUnion": IntersectionOverUnion,
    "oks": ObjectKeypointSimilarity,
    "ObjectKeypointSimilarity": ObjectKeypointSimilarity,
}


def get_similarity_module(name: str) -> Type[SimilarityModule]:
    """Given the name of one of the SimilarityModules, return an instance."""
    return get_instance(instance=name, instances=SIMILARITIES, inst_class=SimilarityModule)


def register_similarity_module(name: str, new_combine: Type[SimilarityModule]) -> None:
    """Register a new similarity module with name in ``SIMILARITIES``, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_combine, instances=SIMILARITIES, inst_class=SimilarityModule)
