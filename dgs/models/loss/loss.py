"""
Custom loss functions.
"""

import torch as t
from torch import nn

from dgs.utils.config import DEF_VAL
from dgs.utils.types import Loss


class CrossEntropyLoss(Loss):
    """Compute the Cross Entropy Loss after computing the LogSoftmax on the input data."""

    def __init__(self, **kwargs):
        super().__init__()
        # self.log_softmax = nn.LogSoftmax(dim=1)
        default_kwargs: dict[str, any] = DEF_VAL["cross_entropy_loss"].copy()
        default_kwargs.update(kwargs)
        self.cross_entropy_loss = nn.CrossEntropyLoss(**default_kwargs)

    def forward(self, inputs: t.Tensor, targets: t.Tensor) -> t.Tensor:
        """Given predictions of shape ``[B x nof_classes]`` and targets of shape ``[B]``
        compute and return the CrossEntropyLoss.
        """
        # inputs = self.log_softmax(inputs)
        return self.cross_entropy_loss(inputs, targets)
