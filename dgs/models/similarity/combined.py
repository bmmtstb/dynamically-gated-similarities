"""
Models that combine the results of two or more similarity matrices.
"""
from abc import abstractmethod

import torch
from torch import nn

from dgs.models.module import BaseModule
from dgs.utils.torchtools import configure_torch_module
from dgs.utils.types import Config, NodePath, Validations

static_alpha_validation: Validations = {"alpha": ["sized"]}


class CombineSimilarityModule(BaseModule, nn.Module):
    """Given two or more similarity matrices, combine them into a single similarity matrix."""

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError


class DynamicallyGatedSimilarities(CombineSimilarityModule):
    r"""Use alpha to weight the two similarity matrices

    Given a weight :math:`\alpha`, compute the weighted similarity between :math:`S_1` and :math:`S_2`
    as :math:`S = \alpha \cdot S_1 + (1 - \alpha) \cdot S_2`.
    Thereby :math:`\alpha` can either be a single float value in :math:`[0, 1]` or a float-tensor of the same shape as
    :math:`S_1` again with values in :math:`[0,1]`.

    Different shapes of :math:`S_1` and :math:`S_2`
    -----------------------------------------------

    It is possible that :math:`S_1` and :math:`S_2` have different shapes in at least one dimension.
    """

    @staticmethod
    def forward(alpha: torch.FloatTensor, s1: torch.FloatTensor, s2: torch.FloatTensor) -> torch.FloatTensor:
        """The forward call of this module combines two weight matrices given a third importance weight :math:`\alpha`.
        :math:`\alpha` describes how important s1 is, while :math:`(1- \alpha)` does the same for s2.

        All tensors should be on the same device and ``s1`` and ``s2`` should have the same shape.

        Args:
            alpha (torch.FloatTensor):
                Weight :math:`\alpha`. Should be a FloatTensor in range [0,1].
                The shape of :math:`\alpha` can either be ``[]``, ``[1 (x 1)]``, or ``[N x 1]``.
            s1 (torch.FloatTensor): A weight matrix as FloatTensor with values in range [0,1] of shape ``[N x T]``.
            s2 (torch.FloatTensor): A weight matrix as FloatTensor with values in range [0,1] of shape ``[N x T]``.

        Returns:
            torch.FloatTensor: The weighted similarity matrix.

        Raises:
            ValueError: If alpha or the matrices have invalid shapes.
        """
        if (a_max := torch.max(alpha)) > 1.0 or torch.min(alpha) < 0.0:
            raise ValueError(f"alpha should lie in the range [0,1], but got [{torch.min(alpha)}, {a_max}]")

        if len(alpha.shape) > 2:
            alpha.squeeze_()
            if len(alpha.shape) >= 2:
                raise ValueError(f"alpha has the wrong shape {alpha.shape}.")

        if len(alpha.shape) == 1 and alpha.shape[0] != 1:
            # [N] -> [N x 1]
            alpha.unsqueeze_(-1)
        elif len(alpha.shape) == 2 and alpha.shape[1] != 1:
            raise ValueError(f"If alpha is two dimensional, the second dimension has to be 1 but got {alpha.shape}.")

        if s1.shape != s2.shape:
            raise ValueError(f"s1 and s2 should have the same shapes, but are: {s1.shape} {s2.shape}.")
        if len(alpha.shape) > 0 and alpha.shape[0] != 1 and alpha.shape[0] != s1.shape[0]:
            raise ValueError(
                f"If the length of the first dimension of alpha is not 1, "
                f"the first dimension has to equal the first dimension of s1 and s2 but got {alpha.shape}."
            )

        return alpha * s1 + (torch.ones_like(alpha) - alpha) * s2


@configure_torch_module
class StaticAlphaWeightingModule(CombineSimilarityModule):
    """Weight two or more similarity matrices using constant (float) values for alpha."""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(static_alpha_validation)

        self.alpha = nn.Parameter(
            torch.tensor(self.params["alpha"], device=self.device).reshape(-1).float(),
            requires_grad=False,
        )

        if not torch.allclose(a_sum := torch.sum(torch.abs(self.alpha)), torch.tensor(1.0)):
            raise ValueError(f"alpha should sum to 1.0, but got {a_sum:.8f}")

    def forward(self, *tensors, **_kwargs) -> torch.FloatTensor:
        """Given alpha from the configuration file and args of the same length,
        multiply each alpha with each matrix and compute the sum.

        Args:
            tensors (tuple[torch.Tensor, ...]): A number of similarity tensors.
                Should have the same length as `alpha`.
                All the tensors should have the same size.

        Returns:
            torch.Tensor: Weighted similarity matrix.

        Raises:
            ValueError: If the ``tensors`` argument has the wrong shape
            TypeError: If the ``tensors`` argument contains an object that is not a `torch.tensor`.
        """
        if any(not isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("All the values in args should be tensors.")
        if len(tensors) != len(self.alpha):
            raise ValueError(
                f"The length of the similarity matrices {len(tensors)} should equal the length of alpha {len(self.alpha)}"
            )
        if len(tensors) > 1 and any(t.shape != tensors[0].shape for t in tensors):
            raise ValueError("The shapes of every tensor should match.")
        return torch.sum(torch.stack([alpha * mat for alpha, mat in zip(self.alpha, tensors)]), dim=0).float()
