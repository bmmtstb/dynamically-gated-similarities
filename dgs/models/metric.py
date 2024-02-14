"""
Methods for handling the computation of distances and other metrics.
"""

import warnings
from typing import Type, Union

import torch
from torch import nn
from torch.linalg import vector_norm

from dgs.utils.types import Metric


@torch.compile
@torch.no_grad()
def compute_cmc(
    distmat: torch.Tensor, query_pids: torch.Tensor, gallery_pids: torch.Tensor, ranks: list[int]
) -> dict[int, float]:
    r"""Compute the cumulative matching characteristics metric.
    It is expected, that the distmat has lower values when the predictions are close.

    Cumulative Matching Characteristics
    -----------------------------------

    For further information see: https://cysu.github.io/open-reid/notes/evaluation_metrics.html.

    Single-gallery-shot means that each gallery identity has only one instance in the query.
    The `single-gallery-shot` CMC top-k accuracy is defined as

    .. math::
       Acc_k = \begin{cases}
          1 & \text{if top-}k\text{ ranked gallery samples contain the query identity} \\
          0 & \text{otherwise}
       \end{cases}

    This represents a shifted step function.
    The final CMC curve is computed by averaging the shifted step functions over all the queries.

    The `multi-gallery-shot` accuracy is not implemented.

    Notes:
        Goal of person ReID:
        For each image in the query, find similar persons within the gallery set.

    Notes:
        This method does not remove "invalid" data.
        E.g., in market1501 iff gallery samples have the same pid and camid as the query.

    Args:
        distmat: (Float)Tensor of shape ``[n_query x n_gallery]`` containing the distances between every item from
            gallery and query.
        query_pids: (Long)Tensor of shape ``[n_query (x 1)]`` containing the query IDs.
        gallery_pids: (Long)Tensor of shape ``[n_gallery (x 1)]``, containing the gallery IDs.
        ranks: List of integers containing the k values used for the evaluation.

    Returns:
        A list containing the float cmc accuracies for each of the k.
    """

    n_query, n_gallery = distmat.shape

    cmcs: dict[int, float] = {}

    query_pids = query_pids.squeeze().unsqueeze(-1).long()
    gallery_pids = gallery_pids.squeeze().long()

    # sort by distance, lowest to highest
    indices = torch.argsort(distmat, dim=1)  # [n_query x n_gallery]
    # with predictions[indices] := sorted predictions
    # obtain a BoolTensor [n_query x max(ranks)] containing whether the r most probable classes equal the query
    most_prob = gallery_pids[indices][:, : min(max(ranks), n_gallery)]
    matches: torch.Tensor = torch.eq(most_prob, query_pids).bool()

    for rank in ranks:
        orig_rank = rank
        if rank >= n_gallery:
            warnings.warn(
                f"Number of gallery samples {n_gallery} is smaller than the max rank {ranks}. Setting rank.",
                UserWarning,
            )
            rank = n_gallery

        cmc = torch.any(matches[:, :rank], dim=1).sum()

        cmcs[orig_rank] = float(cmc.float().item()) / float(n_query)
    return cmcs


@torch.compile
@torch.no_grad()
def compute_accuracy(prediction: torch.Tensor, target: torch.Tensor, topk: list[int] = None) -> dict[int, float]:
    """Compute the accuracies of a predictor over a tuple of ``k``-top predictions.

    Args:
        prediction: prediction matrix with shape ``[B x num_classes]``.
        target: ground truth labels with shape ``[B]``.
        topk: A list containing the number of values to check for the top-k accuracies.
            Default [1].

    Returns:
        The accuracies for each of the ``k``-top predictions.
    """
    if topk is None:
        topk = [1]

    batch_size = target.size(0)

    _, ids = prediction.topk(max(topk))  # [B x max(topk)]

    ids = ids.long()
    target = target.long()

    correct: torch.BoolTensor = ids.eq(target.view(-1, 1)).bool()  # [B x max(topk)]
    del ids, target

    res: dict[int, float] = {}
    for k in topk:
        acc = correct[:, :k].count_nonzero().mul_(100).double().div_(batch_size)
        res[k] = float(acc.float().item())

    return res


@torch.compile(fullgraph=True)
def custom_cosine_similarity(input1: torch.Tensor, input2: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
    """See https://github.com/pytorch/pytorch/issues/104564#issuecomment-1625348908"""
    # get normalization value
    t1_div = vector_norm(input1, dim=dim, keepdim=True)  # pylint: disable=not-callable
    t2_div = vector_norm(input2, dim=dim, keepdim=True)  # pylint: disable=not-callable

    t1_div = t1_div.clone()
    t2_div = t2_div.clone()
    with torch.no_grad():
        t1_div.clamp_(eps)
        t2_div.clamp_(eps)

    # normalize, avoiding division by 0
    t1_norm = input1 / t1_div
    t2_norm = input2 / t2_div

    return torch.mm(t1_norm, t2_norm.T)


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

        return torch.cdist(input1, input2, p=2).square()


class EuclideanDistanceMetric(Metric):
    """Class to compute the Euclidean distance between two matrices."""

    @staticmethod
    def forward(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance between two matrices with a matching second dimension.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)

        return torch.cdist(input1, input2, p=2)


class CosineSimilarityMetric(Metric):
    r"""Class to compute the cosine similarity between two matrices.

    Notes:

        The cosine similarity is defined as:

        .. math::
           \text{cosine similarity} = S_C(A,B) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = -1
        self.eps = 1e-5

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Due to the sheer size of the |PT21| dataset,
        :func:`~torch.nn.CosineSimilarity` does not work due to memory issues.
        See `this issue <https://github.com/pytorch/pytorch/issues/104564#issuecomment-1625348908>_` for more details.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.

        References:
            https://github.com/pytorch/pytorch/issues/104564#issuecomment-1625348908
        """
        _validate_metric_inputs(input1, input2)

        return custom_cosine_similarity(input1, input2, self.dim, self.eps)


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
        self.dim = -1
        self.eps = 1e-5

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)

        cs = custom_cosine_similarity(input1, input2, self.dim, self.eps)
        return torch.ones_like(cs) - cs


METRICS: dict[str, Type[Metric]] = {
    "CosineSimilarity": CosineSimilarityMetric,  # shorthand name
    "CosineSimilarityMetric": CosineSimilarityMetric,
    "CosineDistance": CosineDistanceMetric,  # shorthand name
    "CosineDistanceMetric": CosineDistanceMetric,
    "EuclideanSquare": EuclideanSquareMetric,  # shorthand name
    "EuclideanSquareMetric": EuclideanSquareMetric,
    "EuclideanDistance": EuclideanDistanceMetric,  # shorthand name
    "EuclideanDistanceMetric": EuclideanDistanceMetric,
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
            f"The given metric '{metric_name}' already exists, please choose another name excluding {METRICS.keys()}."
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
