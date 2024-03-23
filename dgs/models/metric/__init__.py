"""
Load, register, and initialize different metric functions.
"""

from typing import Type

from torch import nn

from dgs.utils.loader import get_instance, register_instance
from dgs.utils.types import Instance, Metric
from .metric import (
    CosineDistanceMetric,
    CosineSimilarityMetric,
    EuclideanDistanceMetric,
    EuclideanSquareMetric,
    IOUDistance,
    NegativeSoftmaxEuclideanDistance,
    NegativeSoftmaxEuclideanSquaredDistance,
    PairwiseDistanceMetric,
    TorchreidCosineDistance,
    TorchreidEuclideanSquaredDistance,
)

METRICS: dict[str, Type[Metric]] = {
    "CosineSimilarity": CosineSimilarityMetric,  # alias
    "CosineSimilarityMetric": CosineSimilarityMetric,
    "CosineDistance": CosineDistanceMetric,  # alias
    "CosineDistanceMetric": CosineDistanceMetric,
    "EuclideanSquare": EuclideanSquareMetric,  # alias
    "EuclideanSquareMetric": EuclideanSquareMetric,
    "EuclideanDistance": EuclideanDistanceMetric,  # alias
    "EuclideanDistanceMetric": EuclideanDistanceMetric,
    "PairwiseDistance": PairwiseDistanceMetric,  # alias
    "PairwiseDistanceMetric": PairwiseDistanceMetric,
    "NegSoftmaxEuclideanDist": NegativeSoftmaxEuclideanDistance,  # alias
    "NegativeSoftmaxEuclideanDistance": NegativeSoftmaxEuclideanDistance,
    "NegSoftmaxEuclideanSqDist": NegativeSoftmaxEuclideanSquaredDistance,  # alias
    "NegativeSoftmaxEuclideanSquaredDistance": NegativeSoftmaxEuclideanSquaredDistance,
    "IoUDistance": IOUDistance,
    "IOUDistance": IOUDistance,
    "TorchPairwiseDistance": nn.PairwiseDistance,
    "TorchCosineSimilarity": nn.CosineSimilarity,
    "TorchreidEuclideanSquaredDistance": TorchreidEuclideanSquaredDistance,
    "TorchreidCosineDistance": TorchreidCosineDistance,
}


def register_metric(name: str, new_metric: Type[Metric]) -> None:
    """Register a new metric to be used with custom configs.

    Args:
        name: Name of the new metric, e.g. "CustomDistance".
            The name cannot be a value that is already in :data:``METRICS``.
        new_metric: The type of metric to register.

    Raises:
        ValueError: If ``metric_name`` is in ``METRICS.keys()`` or the ``metric`` is invalid.

    Examples::

        import torch
        from torch import nn
        class CustomDistance(Metric):
            def __init__(...):
                ...
            def forward(self, input: torch.Tensor, target: torch.Tensor):
                return ...
        register_metric("CustomDistance", CustomDistance)
    """
    register_instance(name=name, instance=new_metric, instances=METRICS, inst_class=Metric)


def get_metric(instance: Instance) -> Type[Metric]:
    """Given the name or an instance of a metric, return the respective instance.

    Args:
        instance: Either the name of the metric, which has to be in :data:``METRICS``, or a subclass of ``Metric``.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given metric.
    """
    return get_instance(instance=instance, instances=METRICS, inst_class=Metric)
