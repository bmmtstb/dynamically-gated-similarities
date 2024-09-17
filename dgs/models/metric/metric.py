"""
Methods for handling the computation of distances and other metrics.
"""

import warnings

import torch as t
from torch import nn
from torch.linalg import vector_norm
from torch.nn import PairwiseDistance
from torchvision import tv_tensors as tvte
from torchvision.ops import box_iou
from torchvision.transforms.v2 import ConvertBoundingBoxFormat

from dgs.utils.types import Metric

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Cython evaluation.*is unavailable", category=UserWarning)
    try:
        # If torchreid is installed using `./dependencies/torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.metrics.distance import (
            euclidean_squared_distance as TorchreidESD,
            cosine_distance as TorchreidCD,
        )
    except ModuleNotFoundError:
        # if torchreid is installed using `pip install torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.reid.metrics.distance import (
            euclidean_squared_distance as TorchreidESD,
            cosine_distance as TorchreidCD,
        )


def compute_cmc(distmat: t.Tensor, query_pids: t.Tensor, gallery_pids: t.Tensor, ranks: list[int]) -> dict[int, float]:
    r"""Compute the cumulative matching characteristics metric.
    It is expected that the distmat has lower values when the predictions are close.

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
    indices = t.argsort(distmat, dim=1)  # [n_query x n_gallery]
    # with predictions[indices] := sorted predictions
    # obtain a BoolTensor [n_query x max(ranks)] containing whether the r most probable classes equal the query
    most_prob = gallery_pids[indices][:, : min(max(ranks), n_gallery)]
    matches: t.Tensor = t.eq(most_prob, query_pids).bool()

    for rank in ranks:
        orig_rank = rank
        if rank >= n_gallery:
            warnings.warn(
                f"Number of gallery samples {n_gallery} is smaller than the max rank {ranks}. Setting rank.",
                UserWarning,
            )
            rank = n_gallery

        cmc = t.any(matches[:, :rank], dim=1).sum()

        cmcs[orig_rank] = float(cmc.float().item()) / float(n_query)
    return cmcs


def compute_accuracy(prediction: t.Tensor, target: t.Tensor, topk: list[int] = None) -> dict[int, float]:
    """Compute the accuracies of a predictor over a tuple of ``k``-top predictions.
    Will use the k-biggest values in prediction.

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

    _, ids = prediction.topk(k=max(topk))  # [B x max(topk)]

    ids = ids.long()
    target = target.long()

    correct: t.Tensor = ids.eq(target.view(-1, 1)).bool()  # [B x max(topk)]

    res: dict[int, float] = {}
    for k in topk:
        acc = correct[:, :k].any(dim=1).sum().double().mul_(100.0 / float(batch_size))
        res[k] = float(acc.float().item())

    return res


def compute_near_k_accuracy(a_pred: t.Tensor, a_targ: t.Tensor, ks: list[int]) -> dict[int, float]:
    r"""Compute the number of correct predictions within a margin of k percent for all k.

    Test whether the predicted alpha probability (:math:`\alpha_{\mathrm{pred}}`)
    matches the given ground truth probability (:math:`\alpha_{\mathrm{correct}}`).
    With :math:`\alpha{\mathrm{pred}} = \frac{\alpha_{\mathrm{nof correct}}}{\mathrm{nof total}}`,
    :math`\alpha{\mathrm{pred}}` is counted as correct if
    :math:`\alpha{\mathrm{pred}}-k \leq \alpha{\mathrm{correct}} \leq \alpha{\mathrm{pred}}+k`.

    Args:
        a_pred: The predicted alpha probabilities as tensor of shape ``[N (x 1)]``.
        a_targ: The correct / target alpha probabilities as tensor of shape ``[N (x 1)]``.
        ks: A list of length ``K`` containing percentage values.
            Used to check whether the accuracies lie within a margin of k percent.

    Returns:
        A dict mapping the integer value ``k`` to the respective accuracy.
    """
    if a_pred.ndim > 2 or a_targ.ndim > 2:
        raise NotImplementedError
    if a_pred.ndim == 2 and a_pred.size(-1) != 1:
        raise ValueError(f"Alpha pred should be one dimensional. Got: {a_pred.shape}")
    if a_targ.ndim == 2 and a_targ.size(-1) != 1:
        raise ValueError(f"Alpha target should be one dimensional. Got: {a_targ.shape}")
    N = len(a_pred)
    if len(a_targ) != N:
        raise ValueError(f"alpha_pred and alpha_targ must have the same length, got {N} and {len(a_targ)}.")
    if any(k < 0 for k in ks):
        raise ValueError(f"ks must be positive, got {ks}.")
    k_float = t.tensor(ks, dtype=t.float32, device=a_pred.device).reshape(1, -1) / 100.0  # -> [1 x K]
    # make sure pred and target are 2d
    if a_pred.ndim == 1:
        a_pred = a_pred.unsqueeze(-1)  # -> [N x 1]
    if a_targ.ndim == 1:
        a_targ = a_targ.unsqueeze(-1)  # -> [N x 1]
    correct = t.bitwise_and((a_pred - k_float) <= a_targ, a_targ <= (a_pred + k_float + 1e-7)).sum(dim=0)  # -> [N]
    accuracies: list[float] = (correct / N).tolist()
    return dict(zip(ks, accuracies))


def custom_cosine_similarity(input1: t.Tensor, input2: t.Tensor, dim: int, eps: float) -> t.Tensor:
    """See https://github.com/pytorch/pytorch/issues/104564#issuecomment-1625348908"""
    # get normalization value
    t1_div = vector_norm(input1, dim=dim, keepdim=True)  # pylint: disable=not-callable
    t2_div = vector_norm(input2, dim=dim, keepdim=True)  # pylint: disable=not-callable

    t1_div = t1_div.clone()
    t2_div = t2_div.clone()
    with t.no_grad():
        t1_div.clamp_(eps)
        t2_div.clamp_(eps)

    # normalize, avoiding division by 0
    t1_norm = input1 / t1_div
    t2_norm = input2 / t2_div

    return t.mm(t1_norm, t2_norm.T)


def _validate_metric_inputs(input1: t.Tensor, input2: t.Tensor) -> None:
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
    def forward(input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        """Compute squared Euclidean distance between two matrices with a matching second dimension.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)

        return t.cdist(input1, input2, p=2).square()


class EuclideanDistanceMetric(Metric):
    """Class to compute the Euclidean distance between two matrices."""

    @staticmethod
    def forward(input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        """Compute Euclidean distance between two matrices with a matching second dimension.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)

        return t.cdist(input1, input2, p=2)


class CosineSimilarityMetric(Metric):
    r"""Class to compute the cosine similarity between two matrices.

    Notes:

        The cosine similarity is defined as:

        .. math::
           \text{cosine similarity} = S_C(A,B) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
    """

    def __init__(self, *args, **kwargs) -> None:
        self.dim = kwargs.pop("dim", -1)
        self.eps = kwargs.pop("eps", 1e-5)
        super().__init__(*args, **kwargs)

    def forward(self, input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
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

    def __init__(self, *args, **kwargs) -> None:
        self.dim = kwargs.pop("dim", -1)
        self.eps = kwargs.pop("eps", 1e-5)
        super().__init__(*args, **kwargs)

    def forward(self, input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        """

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            tensor of shape ``[a x b]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)

        cs = custom_cosine_similarity(input1, input2, self.dim, self.eps)
        return t.ones_like(cs) - cs


class PairwiseDistanceMetric(Metric):
    """Class to compute the pairwise distance. For more details see :class:`torch.nn.PairwiseDistance`."""

    def __init__(self, *args, **kwargs):

        p = kwargs.pop("p", 2)
        eps = kwargs.pop("eps", 1e-5)
        keepdim = kwargs.pop("keepdim", False)

        super().__init__(*args, **kwargs)
        dist = PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        self.register_module("dist", dist)

    def forward(self, input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        """Compute the pairwise distance between the two inputs, where the second dimension has to match.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[a x E]``, has to have the same shape as input1.

        Returns:
            tensor of shape ``[a (x 1)]`` containing the distances.
        """
        _validate_metric_inputs(input1, input2)
        return self.dist(input1, input2)


class NegativeSoftmaxEuclideanDistance(Metric):
    """Class to compute the Softmax distribution of the negative Euclidean distance.

    Keyword Args:
        softmax_dim (int): The dimension along which to compute the softmax.
    """

    def __init__(self, *args, softmax_dim: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist = EuclideanDistanceMetric()
        self.softmax = nn.Softmax(dim=softmax_dim)

    def forward(self, input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        """First compute the Euclidean distance between the two inputs, of which the second dimension has to match.
        Then compute the softmax of the negative distance along the second dimension.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            A tensor of shape ``[a x b]`` containing the similarity between the inputs as probability.
            By default, the softmax is computed along the last dimension,
            but you can change the behavior by changing the kwargs during initialization.
        """
        _validate_metric_inputs(input1, input2)
        d = self.dist(input1, input2)
        return self.softmax(t.neg(d))


class NegativeSoftmaxEuclideanSquaredDistance(Metric):
    """Class to compute the Softmax distribution of the negative squared Euclidean distance.

    Keyword Args:
        softmax_dim (int): The dimension along which to compute the softmax.
    """

    def __init__(self, *args, softmax_dim: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist = EuclideanSquareMetric()
        self.softmax = nn.Softmax(dim=softmax_dim)

    def forward(self, input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        """First compute the squared Euclidean distance between the two inputs.
        The second dimension of the inputs has to match.
        Then compute the softmax of the negative distance along the second dimension.

        Args:
            input1: tensor of shape ``[a x E]``
            input2: tensor of shape ``[b x E]``

        Returns:
            A tensor of shape ``[a x b]`` containing the similarity between the inputs as probability.
            By default, the softmax is computed along the last dimension,
            but you can change the behavior by changing the kwargs during initialization.
        """
        _validate_metric_inputs(input1, input2)
        d = self.dist(input1, input2)
        return self.softmax(t.neg(d))


class IOUDistance(Metric):
    """Class to compute the intersection-over-union distance.

    Defined as :math:`d=1-IoU`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = ConvertBoundingBoxFormat("XYXY")

    def forward(self, input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        """Compute the intersection-over-union between two input tensors.

        The inputs should be :class:`tv_tensors.BoundingBoxes`.
        This function will transform given bounding boxes to 'XYXY' if they have a different format.

        Args:
            input1: bbox of shape ``[a x 4]``
            input2: bbox of shape ``[b x 4]``

        Returns:
            A tensor of shape ``[a x b]`` containing the distances between the inputs.

        Raises:
            TypeError: If input1 or input2 is not a :class:`tv_tensors.BoundingBoxes` object.
        """
        if isinstance(input1, tvte.BoundingBoxes):
            if input1.format != tvte.BoundingBoxFormat.XYXY:
                input1 = self.transform(input1)
        else:
            raise TypeError(f"input1 should be an instance of tv_tensors.BoundingBoxes, but got {type(input1)}.")

        if isinstance(input2, tvte.BoundingBoxes):
            if input2.format != tvte.BoundingBoxFormat.XYXY:
                input2 = self.transform(input2)
        else:
            raise TypeError(f"input2 should be an instance of tv_tensors.BoundingBoxes, but got {type(input2)}.")

        iou: t.Tensor = box_iou(input1, input2)

        return t.ones_like(iou) - iou


class TorchreidEuclideanSquaredDistance(Metric):
    """Call TorchReid's version of the Euclidean squared distance."""

    @staticmethod
    def forward(input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        _validate_metric_inputs(input1, input2)
        return TorchreidESD(input1=input1, input2=input2)


class TorchreidCosineDistance(Metric):
    """Call TorchReid's version of the cosine distance."""

    @staticmethod
    def forward(input1: t.Tensor, input2: t.Tensor) -> t.Tensor:
        _validate_metric_inputs(input1, input2)
        return TorchreidCD(input1=input1, input2=input2)
