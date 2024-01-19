"""
Modules for handling similarity functions or other models that return similarity scores between two (or more) inputs.
"""
from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .combined import CombineSimilarityModule, DynamicallyGatedSimilarities, StaticAlphaWeightingModule
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


def get_combined_similarity_module(name: str) -> Type[CombineSimilarityModule]:
    """Given the name of one module that combines different similarity modules, return an instance."""
    modules: dict[str, Type[CombineSimilarityModule]] = {
        "DGS": DynamicallyGatedSimilarities,
        "dynamic_alpha": DynamicallyGatedSimilarities,  # synonym for DGS
        "static_alpha": StaticAlphaWeightingModule,
    }
    if name not in modules:
        raise InvalidParameterException(f"Unknown combined similarity module with name: {name}.")
    return modules[name]
