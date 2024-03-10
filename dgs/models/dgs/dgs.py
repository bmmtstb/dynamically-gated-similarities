"""
Base class for a torch module that contains the heart of the dynamically gated similarity tracker.
"""

import torch
from torch import nn

from dgs.models.combine import CombineSimilaritiesModule, get_combine_module
from dgs.models.module import BaseModule
from dgs.models.similarity import get_similarity_module
from dgs.utils.config import get_sub_config
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

dgs_validations: Validations = {
    "names": [list, ("forall", str)],
    "combine": [str],
}


class DGSModule(BaseModule, nn.Module):
    """Torch module containing the code for the model called 'dynamically gated similarities'.

    Params
    ------

    names (list[str]):
        The names of the keys in the configuration containing all the wanted :class:`SimilarityModule` s.
    combine (str):
        The name of the key in the configuration containing the parameters for the module to combine the similarities
        (see the parameters at :class:`.CombineSimilaritiesModule`).
    similarity_softmax (bool, optional):
        Whether to compute the softmax of every resulting similarity matrix (independently) before combining them.
        Default False.
    combined_softmax (bool, optional):
        Whether to compute the softmax after the similarities have been summed up / combined.
        Default False
    """

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(dgs_validations)

        # list of the modules computing the similarities
        names = self.params["names"]
        self.sim_mods = nn.ModuleList(
            [
                get_similarity_module(get_sub_config(config=config, path=[k])["module_name"])(config=config, path=[k])
                for k in names
            ]
        )

        # if wanted, compute the softmax of every resulting similarity matrix
        self.similarity_softmax = nn.Sequential()
        if self.params.get("similarity_softmax", False):
            self.similarity_softmax.append(nn.Softmax(dim=-1))

        # module for combining multiple similarities
        combine_name = self.params["combine"]
        combine: CombineSimilaritiesModule = get_combine_module(
            name=get_sub_config(config=config, path=[combine_name])["module_name"]
        )(config=config, path=[combine_name])
        self.register_module(name="combine", module=combine)

        # if wanted, compute the softmax after the similarities have been summed up / combined
        self.combined_softmax = nn.Sequential()
        if self.params.get("combined_softmax", False):
            self.combined_softmax.append(nn.Softmax(dim=-1))

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        return self.forward(*args, **kwargs)

    def forward(self, ds: State, target: State) -> torch.Tensor:
        """Given a State containing the current detections and a target, compute the similarity between every pair.

        Returns:
            The combined similarity matrix as tensor of shape ``[nof_detections x (nof_tracks + 1)]``.
        """
        nof_det = len(ds)

        # compute similarity for every module and possibly compute the softmax
        results = [self.similarity_softmax(m(ds, target)) for m in self.sim_mods]

        # combine and possibly compute softmax
        combined: torch.Tensor = self.combined_softmax(self.combine(*results))

        # add column for the empty / new track
        new_track = torch.zeros((nof_det, 1), dtype=self.precision, device=self.device)
        return torch.cat([combined, new_track], dim=-1)
