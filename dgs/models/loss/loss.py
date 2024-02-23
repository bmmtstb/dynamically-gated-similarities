"""
Custom loss functions.
"""

import torch
from torch import nn

from dgs.utils.types import Loss


class CrossEntropyLoss(Loss):
    """Compute the Cross Entropy Loss after computing the LogSoftmax on the input data."""

    def __init__(self, **kwargs):
        super().__init__()
        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, inputs: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        """Given predictions of shape ``[B x nof_classes]`` and targets of shape ``[B]``
        compute and return the CrossEntropyLoss.
        """
        # inputs = self.log_softmax(inputs)
        return self.cross_entropy_loss(inputs, targets)
