"""
Implementation of modules that combine two or more similarity matrices.
Obtain similarity matrices as a result of one or multiple
:class:``~dgs.models.similarity.similarity.SimilarityModule`` s.
"""

from abc import abstractmethod

import torch
from torch import nn

from dgs.models.module import BaseModule
from dgs.utils.torchtools import configure_torch_module
from dgs.utils.types import Config, NodePath, Validations

combine_validations: Validations = {
    "module_name": [str],
}

static_alpha_validation: Validations = {
    "alpha": [
        list,
        ("longer eq", 1),  # there is  actually no need for combining a single model
        ("forall", [float, ("within", (0.0, 1.0))]),
        lambda x: abs(sum(x_i for x_i in x) - 1.0) < 1e-6,  # has to sum to 1
    ],
}


class CombineSimilaritiesModule(BaseModule, nn.Module):
    """Given two or more similarity matrices, combine them into a single similarity matrix."""

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(combine_validations)

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


@configure_torch_module
class DynamicallyGatedSimilarities(CombineSimilaritiesModule):
    r"""Use alpha to weight the two similarity matrices

    Given a weight :math:`\alpha`, compute the weighted similarity between :math:`S_1` and :math:`S_2`
    as :math:`S = \alpha \cdot S_1 + (1 - \alpha) \cdot S_2`.
    Thereby :math:`\alpha` can either be a single float value in :math:`[0, 1]` or a float-tensor of the same shape as
    :math:`S_1` again with values in :math:`[0,1]`.

    Different shapes of :math:`S_1` and :math:`S_2`
    -----------------------------------------------

    It is possible that :math:`S_1` and :math:`S_2` have different shapes in at least one dimension.
    """

    def forward(self, *tensors, alpha: torch.Tensor = torch.tensor([0.5, 0.5]), **_kwargs) -> torch.Tensor:
        """The forward call of this module combines two weight matrices given a third importance weight :math:`\alpha`.
        :math:`\alpha` describes how important s1 is, while :math:`(1- \alpha)` does the same for s2.

        All tensors should be on the same device and ``s1`` and ``s2`` should have the same shape.

        Args:
            tensors (tuple[torch.Tensor, ...]): Two weight matrices as tuple of FloatTensors.
                Both should have values in range [0,1] and be of the same shape ``[N x T]``.
            alpha: Weight :math:`\alpha`. Should be a FloatTensor in range [0,1].
                The shape of :math:`\alpha` can either be ``[]``, ``[1 (x 1)]``, or ``[N x 1]``.

        Returns:
            torch.Tensor: The weighted similarity matrix.

        Raises:
            ValueError: If alpha or the matrices have invalid shapes.
        """
        if len(tensors) != 2:
            raise ValueError(f"There should be exactly two matrices in the tensors argument, got {len(tensors)}")
        if any(not isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("All matrices should be torch (float) tensors.")
        s1, s2 = tensors
        if (a_max := torch.max(alpha)) > 1.0 or torch.min(alpha) < 0.0:
            raise ValueError(f"alpha should lie in the range [0,1], but got [{torch.min(alpha)}, {a_max}]")

        if alpha.ndim > 2:
            alpha.squeeze_()
            if alpha.ndim >= 2:
                raise ValueError(f"alpha has the wrong shape {alpha.shape}.")

        if alpha.ndim == 1 and alpha.shape[0] != 1:
            # [N] -> [N x 1]
            alpha.unsqueeze_(-1)
        elif alpha.ndim == 2 and alpha.shape[1] != 1:
            raise ValueError(f"If alpha is two dimensional, the second dimension has to be 1 but got {alpha.shape}.")

        if s1.shape != s2.shape:
            raise ValueError(f"s1 and s2 should have the same shapes, but are: {s1.shape} {s2.shape}.")
        if alpha.ndim > 0 and alpha.shape[0] != 1 and alpha.shape[0] != s1.shape[0]:
            raise ValueError(
                f"If the length of the first dimension of alpha is not 1, "
                f"the first dimension has to equal the first dimension of s1 and s2 but got {alpha.shape}."
            )

        return alpha * s1 + (torch.ones_like(alpha) - alpha) * s2


@configure_torch_module
class StaticAlphaCombine(CombineSimilaritiesModule):
    """
    Weight two or more similarity matrices using constant (float) values for alpha.

    Params
    ------

    alpha (list[float]):
        A list containing the constant weights for the different similarities.
        The weights should be probabilities and therefore sum to one and lie within [0..1].

    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(static_alpha_validation)

        alpha = torch.tensor(self.params["alpha"], dtype=self.precision).reshape(-1)
        self.register_buffer("alpha_const", alpha)
        self.len_alpha: int = len(alpha)

        if not torch.allclose(a_sum := torch.sum(torch.abs(alpha)), torch.tensor(1.0)):  # pragma: no cover  # redundant
            raise ValueError(f"alpha should sum to 1.0, but got {a_sum:.8f}")

    def forward(self, *tensors, **_kwargs) -> torch.Tensor:
        """Given alpha from the configuration file and args of the same length,
        multiply each alpha with each matrix and compute the sum.

        Args:
            tensors (tuple[torch.Tensor, ...]): A number of similarity tensors.
                Should have the same length as `alpha`.
                All the tensors should have the same size.

        Returns:
            The weighted similarity matrix as FloatTensor.

        Raises:
            ValueError: If the ``tensors`` argument has the wrong shape
            TypeError: If the ``tensors`` argument contains an object that is not a `torch.tensor`.
        """
        if not isinstance(tensors, tuple):
            raise NotImplementedError(
                f"Unknown type for tensors, expected tuple of torch.Tensor but got {type(tensors)}"
            )

        if any(not isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError("All the values in args should be tensors.")

        if len(tensors) > 1 and any(t.shape != tensors[0].shape for t in tensors):
            raise ValueError("The shapes of every tensor should match.")

        if len(tensors) == 1 and self.len_alpha != 1:
            # given a single already stacked tensor or a single valued alpha
            tensors = tensors[0]
        else:
            tensors = torch.stack(tensors)

        if self.len_alpha != 1 and len(tensors) != self.len_alpha:
            raise ValueError(
                f"The length of the tensors {len(tensors)} should equal the length of alpha {self.len_alpha}"
            )

        return torch.tensordot(self.alpha_const, tensors.float(), dims=1)

    def terminate(self) -> None:  # pragma: no cover
        del self.alpha, self.alpha_const, self.len_alpha
