"""
Implementation of modules that use dynamic weights to combine multiple similarities.
"""

import torch as t
from torch import nn

from dgs.models.combine.combine import CombineSimilaritiesModule
from dgs.models.loader import module_loader
from dgs.utils.state import State
from dgs.utils.torchtools import configure_torch_module
from dgs.utils.types import Config, NodePath, Validations

alpha_combine_validation: Validations = {}

dynamic_alpha_validation: Validations = {
    "alpha_modules": ["NodePaths", ("longer eq", 1)],
}


@configure_torch_module
class DynamicAlphaCombine(CombineSimilaritiesModule):
    r"""Use inputs and multiple alpha modules to weight the similarity matrices.

    Notes:
        The models for computing the per-similarity alpha has to be set manually after the initialization.

    Given ``N`` inputs to the alpha module (e.g. the visual embeddings of ``N`` images,
    or ``N`` different sized inputs like the bbox, pose, and visual embedding of a single crop),
    compute the alpha weights for the similarity matrices.
    Then use :math:`\alpha_i` to compute the weighted sum of all the similarity matrices  :math:`S_i`
    as :math:`S = \sum_N \alpha_i \cdot S_i`.

    Every :math:`\alpha_i` can either be a single float value in range :math:`[0, 1]` or
    a (float-) tensor of the same shape as :math:`S_i` again with values in :math:`[0,1]`.

    Params
    ------

    alpha_modules (list[NodePath]):
        A list containing paths to multiple :class:`BaseAlphaModule` s.

    Optional Params
    ---------------

    """

    alpha_models: nn.ModuleList
    """The model that computes the alpha values from given inputs."""

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config, path)
        self.validate_params(dynamic_alpha_validation)

        self.alpha_models = nn.ModuleList(
            [module_loader(config=config, module_type="alpha", key=path) for path in self.params["alpha_modules"]]
        ).to(device=self.device)

    def forward(
        self,
        *tensors: t.Tensor,
        s: State = None,
        **_kwargs,
    ) -> t.Tensor:
        r"""The forward call of this module combines an arbitrary number of similarity matrices
        using an importance weight :math:`\alpha`.

        :math:`\alpha_i` describes how important :math:`s_i` is.
        The sum of all :math:`\alpha_i` should be 1 by definition given the last layer is a softmax layer.
        :math:`\alpha` is computed using this class' neural network and the given ``alpha_input`` tensor.

        All tensors should be on the same device and all :math:`s_i` should have the same shape.

        Args:
            tensors: A tuple of tensors describing similarities between the detections and tracks.
                All ``S`` similarity matrices of this iterable should have values in range ``[0,1]``,
                be of the same shape ``[D x T]``, and be on the same device.
                If ``tensors`` is a single tensor, it should have the shape ``[S x D x T]``.
                ``S`` can be any number of similarity matrices greater than 0,
                even though only values greater than 1 really make sense.
            s: A :class:`State` containing the batched input data for the alpha models.
                The state should be on the same device as ``tensors``.

        Returns:
            torch.Tensor: The weighted similarity matrix as tensor of shape ``[D x T]``.

        Raises:
            ValueError: If alpha or the matrices have invalid shapes.
            RuntimeError: If one of the tensors is not on the correct device.
            TypeError: If one of the tensors or one of the alpha inputs is not of type class:`torch.Tensor`.
        """
        # pylint: disable=too-many-branches

        # validate tensors
        if not isinstance(tensors, tuple):  # pragma: no cover # redundancy
            raise TypeError("tensors should be a tuple containing (float) tensors.")
        if any(not isinstance(tensor, t.Tensor) for tensor in tensors):
            raise TypeError("All similarity matrices should be (float) tensors.")
        if any(tensor.shape != tensors[0].shape for tensor in tensors):
            raise ValueError("All similarity matrices should have the same shape.")
        if any(tensor.device != tensors[0].device for tensor in tensors):
            raise RuntimeError("All tensors should be on the same device.")
        if len(self.alpha_models) != len(tensors):
            raise ValueError(
                f"There should be as many alpha models {len(self.alpha_models)} as tensors {len(tensors)}."
            )
        # force tensors to be shape [S x D x T]
        if len(tensors) == 1:
            tensors = tensors[0]
        else:
            tensors = t.stack(tensors, dim=-3)

        if tensors.ndim != 3:
            raise ValueError(f"Expected a 3D tensor [S x D x T], but got a tensor with shape {tensors.shape}")

        # validate alpha inputs
        if s is None:
            raise ValueError("The state should be given.")
        if not isinstance(s, State):
            raise TypeError(f"s should be a State. Got: {s}")
        if s.device != tensors.device:
            raise RuntimeError("s should be on the same device as tensors.")
        if tensors.size(-2) != s.B:
            raise ValueError(f"The states batch size ({s.B}) should equal D of tensors ({tensors.size(-2)}).")

        # [D x S] with softmax over S dimension
        alpha = nn.functional.softmax(
            t.cat([self.alpha_models[i].forward(s) for i in range(len(tensors))], dim=1), dim=-1
        )

        # [S x D x 1] hadamard [S x D x T] -> [S x D x T] -> sum over all S => [D x T]
        s = t.mul(alpha.T.unsqueeze(-1), tensors).sum(dim=0)

        return s

    def terminate(self) -> None:  # pragma: no cover
        if hasattr(self, "alpha_models"):
            del self.alpha_models


@configure_torch_module
class AlphaCombine(CombineSimilaritiesModule):
    r"""Compute a weighted sum of multiple given similarity matrices and given alpha weights.

    More precisely, given a similarity matrix / tensor  with shape ``[N x T]``,
    and one alpha value per similarity, compute the weighted sum of all the similarity matrices.
    The module will make sure, that :math:`\sum_N \alpha_i = 1`.

    Params
    ------

    Optional Params
    ---------------

    """

    def __init__(self, config: Config, path: NodePath) -> None:
        super().__init__(config, path)
        self.validate_params(alpha_combine_validation)

    def forward(self, *tensors: t.Tensor, alpha: t.Tensor = None, **_kwargs) -> t.Tensor:
        r"""The forward call of this module combines an arbitrary number of similarity matrices
        using an importance weight :math:`\alpha`.

        Args:
            tensors: ``N`` similarity matrices as a tuple of tensors.
                All tensors should have values in range ``[0,1]``, be of the same shape ``[D x T]``,
                and be on the same device.
            alpha: A tensor containing weights in range ``[0,1]``.
                Alpha can have one of the following shapes: ``[N]`` or ``[N x D]``.
                The alpha tensor should be on the same device as the other tensors.

        Returns:
            torch.Tensor: The weighted similarity matrix.

        Raises:
            ValueError: If alpha or the matrices have invalid shapes.
            RuntimeError: If the tensors are not on the same device.
            TypeError: If one of the tensors or alpha is not of type class:`torch.Tensor`.
        """
        # test tensors
        if not isinstance(tensors, tuple):
            raise TypeError(f"tensors should be a tuple containing (float) tensors, got {type(tensors)}.")
        if any(not isinstance(tensor, t.Tensor) for tensor in tensors):
            raise TypeError("All similarity matrices should be (float) tensors.")
        if any(tensor.device != tensors[0].device for tensor in tensors):
            raise RuntimeError("All tensors should be on the same device.")
        tensors = t.stack(tensors)  # [N x D x T]

        # test alpha
        if alpha is None:
            raise ValueError("Alpha should be given.")
        if not isinstance(alpha, t.Tensor):
            raise TypeError("alpha should be a (float) tensor.")
        # test combined
        if len(alpha) != len(tensors):
            raise ValueError(f"Alpha {len(alpha)} should have the same length as the tensors {len(tensors)}.")
        if alpha.device != tensors.device:
            raise RuntimeError("alpha should be on the same device as the tensors.")

        if alpha.ndim == 2 and alpha.shape[-1] != tensors.shape[-2]:
            raise ValueError(f"alpha should have shape [N x D], but got {alpha.shape}")
        if alpha.ndim == 3 and (alpha.shape[-2] != tensors.shape[-2] or alpha.shape[-1] != tensors.shape[-1]):
            raise ValueError(f"alpha should have shape [N x D x T], but got {alpha.shape}")

        if alpha.ndim == 1:
            s = t.tensordot(alpha, tensors, dims=([0], [0]))  # [N] dot [N x D x T] -> [D x T]
        else:
            # fixme add dims for [NxD] and [NxDxT]
            raise NotImplementedError("Alpha with shape [N x D] or [N x D x T] is not yet implemented.")

        return self.softmax(s)
