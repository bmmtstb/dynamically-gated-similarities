"""
Base class for a torch module that contains the heart of the dynamically gated similarity tracker.
"""

import torch
from torch import nn

from dgs.models import BaseModule
from dgs.utils.states import DataSample


class DGSModule(BaseModule, nn.Module):
    """Torch module containing the code for the model called "dynamically gated similarities"."""

    def __call__(self, *args, **kwargs) -> any:
        pass

    def forward(self, ds: DataSample, target: any) -> torch.Tensor:
        """Given a DataSample containing the current detections and a target, compute the similarity between every pair.

        Returns:
            The combined similarity matrix as tensor of shape ``[nof_detections x (nof_tracks + 1)]``.
        """
