"""
Implementation of modules that use static alpha values to combine the similarities.
"""

import torch as t

from dgs.models.combine.combine import CombineSimilaritiesModule
from dgs.utils.torchtools import configure_torch_module
from dgs.utils.types import Config, NodePath, Validations

static_alpha_validation: Validations = {
    "alpha": [
        list,
        ("longer eq", 1),  # there is  actually no need for combining a single model
        ("forall", [float, ("within", (0.0, 1.0))]),
        lambda x: abs(sum(x_i for x_i in x) - 1.0) < 1e-6,  # has to sum to 1
    ],
}


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

        alpha = t.tensor(self.params["alpha"], dtype=self.precision).reshape(-1)
        self.register_buffer("alpha_const", alpha)
        self.len_alpha: int = len(alpha)

        if not t.allclose(a_sum := t.sum(t.abs(alpha)), t.tensor(1.0)):  # pragma: no cover  # redundant
            raise ValueError(f"alpha should sum to 1.0, but got {a_sum:.8f}")

    def forward(self, *tensors, **_kwargs) -> t.Tensor:
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

        if any(not isinstance(tensor, t.Tensor) for tensor in tensors):
            raise TypeError("All the values in args should be tensors.")

        if len(tensors) > 1 and any(tensor.shape != tensors[0].shape for tensor in tensors):
            raise ValueError("The shapes of every tensor should match.")

        if len(tensors) == 1 and self.len_alpha != 1:
            # given a single already stacked tensor or a single valued alpha
            tensors = tensors[0]
        else:
            tensors = t.stack(tensors)

        if self.len_alpha != 1 and len(tensors) != self.len_alpha:
            raise ValueError(
                f"The length of the tensors {len(tensors)} should equal the length of alpha {self.len_alpha}"
            )

        return self.softmax(t.tensordot(self.alpha_const, tensors.float(), dims=1))

    def terminate(self) -> None:  # pragma: no cover
        del self.alpha, self.alpha_const, self.len_alpha
