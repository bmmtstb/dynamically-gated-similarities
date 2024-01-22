"""
Methods for handling the computation of distances and other metrics.
"""
from typing import Type, Union

import torch
from torch import nn

from dgs.utils.types import Metric


def _validate_metric_inputs(input1: torch.Tensor, input2: torch.Tensor) -> None:
    """Metrics should be handed two tensors where the second dimension matches.

    Raises:
        ValueError: If the inputs do not have the right shape.
    """
    if input1.ndim != 2 or input2.ndim != 2 or input1.shape[-1] != input2.shape[-1]:
        raise ValueError(
            f"Inputs must be two-dimensional and the size of the second dimension must match. "
            f"got: {input1.shape} and {input2.shape}."
        )


def _expand_metric_inputs(input1: torch.Tensor, input2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Given two matrices with shapes ``[a x E]`` and ``[b x E]``, expand them both to have shape ``[a x b x E]``"""
    a, b = input1.shape[0], input2.shape[0]
    # make sure i1 and i2 have shape [a,b,E] so we can compute every value of a against every of b
    return input1.unsqueeze_(1).expand(a, b, -1), input2.unsqueeze(0).expand(a, b, -1)


class EuclideanSquareMetric(Metric):
    """Class to compute the squared Euclidean distance between two matrices."""

    @staticmethod
    def forward(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean distance between two matrices with a matching second dimension.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)
        input1, input2 = _expand_metric_inputs(input1, input2)

        return torch.sub(input1, input2).square().sum(dim=-1)


class CosineSimilarityMetric(Metric):
    r"""Class to compute the cosine similarity between two matrices.

    Notes:

        The cosine similarity is defined as:

        .. math::
           \text{cosine similarity} = S_C(A,B) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)
        input1, input2 = _expand_metric_inputs(input1, input2)

        return self.cos(input1, input2)


class CosineDistanceMetric(Metric):
    r"""Class to compute the cosine distance between two matrices.

    Notes:

        The cosine distance is the complement of the cosine similarity in positive space:

        .. math::
           \text{cosine similarity} = S_C(A,B) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}

           \text{cosine distance} = 1 - S_C(A,B)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)
        input1, input2 = _expand_metric_inputs(input1, input2)

        cs = self.cos(input1, input2)
        return torch.ones_like(cs) - cs


METRICS: dict[str, Type[Metric]] = {
    "CosineSimilarity": CosineSimilarityMetric,
    "CosineDistance": CosineDistanceMetric,
    "EuclideanSquare": EuclideanSquareMetric,
    "TorchPairwiseDistance": nn.PairwiseDistance,
    "TorchCosineSimilarity": nn.CosineSimilarity,
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
