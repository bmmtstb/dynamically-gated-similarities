"""
Combine multiple similarity matrices.
Obtain similarity matrices as a result of one or multiple
:class:``~dgs.models.similarity.similarity.SimilarityModule`` s.
"""

from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .combine import (
    CombineSimilaritiesModule,
    DynamicallyGatedSimilarities,
    StaticAlphaCombine,
)


def get_dgs_module(name: str) -> Type[CombineSimilaritiesModule]:
    """Given the name of one module that combines different similarity modules, return an instance."""
    modules: dict[str, Type[CombineSimilaritiesModule]] = {
        "DGS": DynamicallyGatedSimilarities,
        "dynamic_alpha": DynamicallyGatedSimilarities,  # alias for DGS
        "static_alpha": StaticAlphaCombine,
        "constant_alpha": StaticAlphaCombine,  # alias
    }
    if name not in modules:
        raise InvalidParameterException(f"Unknown combine similarities module with name: {name}.")
    return modules[name]
