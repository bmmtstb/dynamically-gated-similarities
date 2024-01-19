"""
Methods for handling the computation of distances and other metrics.
"""
from typing import Type, Union

from torch import nn

from dgs.utils.types import Metric

METRICS: dict[str, Type[Metric]] = {
    "CosineSimilarity": nn.CosineSimilarity,
    "PairwiseDistance": nn.PairwiseDistance,
}


def register_metric(metric_name: str, metric: Type[Metric]) -> None:
    """Register a new metric.

    Args:
        metric_name: Name of the new metric, e.g. "CustomDistance".
            Cannot be a value that is already in `METRICS`.
        metric: The type of metric to register.

    Raises:
        ValueError: If `metric_name` is in `METRICS.keys()` or the `metric` is invalid.

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
    if metric_name in METRICS:
        raise ValueError(
            f"The given name '{metric_name}' already exists, " f"please choose another name excluding {METRICS.keys()}."
        )
    if not (callable(metric) and isinstance(metric, type) and issubclass(metric, Metric)):
        raise ValueError(f"The given metric function is no callable or no subclass of Metric. Got: {metric}")
    METRICS[metric_name] = metric


def get_metric_from_name(name: str) -> Type[Metric]:
    """Given a name of a metric, which is in `METRICS`, return an instance of it.

    Params:
        name: The name of the metric.

    Raises:
        ValueError: If the metric does not exist in `METRICS`.

    Returns:
        An instance of the metric.
    """
    if name not in METRICS:
        raise ValueError(f"Metric '{name}' is not defined.")
    return METRICS[name]


def get_metric(instance: Union[str, callable]) -> Type[Metric]:
    """

    Args:
        instance: Either the name of the metric, which has to be in `METRICS`, or a subclass of `Metric`.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given metric.
    """
    if isinstance(instance, str):
        return get_metric_from_name(instance)
    if isinstance(instance, type) and issubclass(instance, Metric):
        return instance
    raise ValueError(f"Metric '{instance}' is neither string nor subclass of 'Metric'.")
