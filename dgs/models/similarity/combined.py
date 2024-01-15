"""
Models that combine the results of two or more similarity matrices.
"""
import torch

from dgs.models.module import BaseModule


class DynamicallyGatedSimilarities(BaseModule):
    r"""Use alpha to weight the two similarity matrices

    Given a weight :math:`\alpha`, compute the weighted similarity between :math:`S_1` and :math:`S_2`
    as :math:`S = \alpha \cdot S_1 + (1 - \alpha) \cdot S_2`.
    Thereby :math:`\alpha` can either be a single float value in :math:`[0, 1]` or a float-tensor of the same shape as
    :math:`S_1` again with values in :math:`[0,1]`.

    Different shapes of :math:`S_1` and :math:`S_2`
    -----------------------------------------------

    It is possible that :math:`S_1` and :math:`S_2` have different shapes in at least one dimension.

    """

    def __call__(self, *args, **kwargs) -> any:  # noqa
        return self.forward(*args, **kwargs)

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
        if len(alpha.shape) > 2:
            alpha.squeeze_()
            if len(alpha.shape) > 2:
                raise ValueError(f"alpha has the wrong shape {alpha.shape}.")

        if len(alpha.shape) == 1 and alpha.shape[0] != 1:
            # [N] -> [N x 1]
            alpha.unsqueeze_(-1)
        elif len(alpha.shape) == 2 and alpha.shape[1] != 1:
            raise ValueError(f"If alpha is two dimensional, the second dimension has to be 1 but got {alpha.shape}.")

        if s1.shape != s2.shape:
            raise ValueError(f"s1 and s2 should have the same shapes, but are: {s1.shape} {s2.shape}.")
        elif len(alpha.shape) > 0 and alpha.shape[0] != 1 and alpha.shape[0] != s1.shape[0]:
            raise ValueError(
                f"If the length of the first dimension of alpha is not 1, "
                f"the first dimension has to equal the first dimension of s1 and s2 but got {alpha.shape}."
            )

        return alpha * s1 + (torch.ones_like(alpha) - alpha) * s2
