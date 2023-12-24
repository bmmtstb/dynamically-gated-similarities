"""
Modules for handling similarity functions or other models that return similarity scores between two (or more) inputs.
"""
from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .similarity import (
    CosineSimilarityModule,
    DotProductModule,
    EuclideanDistanceModule,
    PairwiseDistanceModule,
    PNormDistanceModule,
    SimilarityModule,
)


def get_similarity_module(name: str) -> Type[SimilarityModule]:
    """Given the name of one of the SimilarityModules, return an instance."""
    modules: dict[str, Type[SimilarityModule]] = {
        "cosine": CosineSimilarityModule,
        "dot": DotProductModule,
        "euclidean": EuclideanDistanceModule,
        "pairwise_dist": PairwiseDistanceModule,
        "p_norm_dist": PNormDistanceModule,
    }
    if name not in modules:
        raise InvalidParameterException(f"Unknown similarity with name: {name}.")

    return modules[name]
