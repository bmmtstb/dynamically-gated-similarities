"""
Base class for a torch module that contains the heart of the dynamically gated similarity tracker.
"""

import torch
from torch import nn

from dgs.models.module import BaseModule
from dgs.models.similarity import get_similarity_module, SIMILARITIES
from dgs.utils.states import DataSample
from dgs.utils.types import Config, NodePath, Validations

dgs_validations: Validations = {
    "names": [list, ("forall", [str, ("in", SIMILARITIES.keys())])],
    # optional
    "paths": ["optional", list, ("forall", [list, ("forall", str)])],
}


class DGSModule(BaseModule, nn.Module):
    """Torch module containing the code for the model called "dynamically gated similarities"."""

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(dgs_validations)

        names = self.params["names"]
        paths: list[NodePath] = self.params.get("paths", [[name] for name in names])
        if len(names) != len(paths):
            raise ValueError(f"Length of paths should equal length of names, but got paths: {paths} and names: {names}")
        self.modules = nn.ModuleList(
            [get_similarity_module(name)(config=config, path=path) for name, path in zip(names, paths)]
        )

    def __call__(self, *args, **kwargs) -> any:
        pass

    def forward(self, ds: DataSample, target: any) -> torch.Tensor:
        """Given a DataSample containing the current detections and a target, compute the similarity between every pair.

        Returns:
            The combined similarity matrix as tensor of shape ``[nof_detections x (nof_tracks + 1)]``.
        """
