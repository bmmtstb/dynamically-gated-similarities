"""
Base class for a torch module that contains the heart of the dynamically gated similarity tracker.
"""

from collections.abc import MutableSequence
from typing import Union

import torch as t
from torch import nn

from dgs.models.combine import get_combine_module
from dgs.models.combine.combine import CombineSimilaritiesModule
from dgs.models.module import BaseModule
from dgs.models.similarity import get_similarity_module
from dgs.models.similarity.similarity import SimilarityModule
from dgs.utils.config import DEF_VAL, get_sub_config
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

dgs_validations: Validations = {
    "names": ["NodePaths"],
    "combine": ["NodePath"],
    # optional
    "new_track_weight": ["optional", float, ("within", (0.0, 1.0))],
}


class DGSModule(BaseModule, nn.Module):
    """Torch module containing the code for the model called 'dynamically gated similarities'.

    Params
    ------

    names (list[NodePath]):
        The names or :class:`NodePath` s of the keys within the current configuration
        which contain all the :class:`.SimilarityModule` s used in this module.
    combine (NodePath):
        The name or :class:`NodePath` of the key in the current configuration containing the parameters for the
        :class:`.CombineSimilaritiesModule` used to combine the similarities.

    Optional Params
    ---------------

    new_track_weight (float, optional):
        The weight of the new tracks as probability.
        "0.0" means, that existing tracks will always be preferred, while "1.0" means that new tracks are preferred.
        Default ``DEF_VAL.dgs.similarity_softmax``.
    """

    sim_mods: Union[nn.ModuleList, MutableSequence[SimilarityModule]]
    combine: CombineSimilaritiesModule
    new_track_weight: t.Tensor

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(dgs_validations)

        # list of the modules computing the similarities
        names: list[NodePath] = self.params["names"]
        self.sim_mods = nn.ModuleList(
            [
                self.configure_torch_module(
                    get_similarity_module(get_sub_config(config=config, path=k)["module_name"])(config=config, path=k),
                )
                for k in names
            ]
        )
        self.configure_torch_module(self.sim_mods)

        # module for combining multiple similarities
        combine_name = self.params["combine"]
        combine: CombineSimilaritiesModule = get_combine_module(
            name=get_sub_config(config=config, path=[combine_name])["module_name"]
        )(config=config, path=[combine_name])
        self.register_module(name="combine", module=self.configure_torch_module(combine))

        # get weight of new tracks
        self.new_track_weight: t.Tensor = t.tensor(
            self.params.get("new_track_weight", DEF_VAL["dgs"]["new_track_weight"]),
            dtype=self.precision,
            device=self.device,
        )

    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        return self.forward(*args, **kwargs)

    def forward(self, ds: State, target: State, **_kwargs) -> t.Tensor:
        """Given a State containing the current detections and a target, compute the similarity between every pair.

        Returns:
            The combined similarity matrix as tensor of shape ``[nof_detections x (nof_tracks + nof_detections)]``.
        """
        nof_det = len(ds)

        # compute similarity for every module and possibly compute the softmax
        results = [m(ds, target) for m in self.sim_mods]

        # combine and possibly compute softmax
        combined: t.Tensor = self.combine(*results, **_kwargs)

        # add a number of columns for the empty / new tracks equal to the length of the input
        # every input should be allowed to get assigned to a new track
        # probability of new tracks can be set through params
        new_track = t.ones((nof_det, nof_det), dtype=self.precision, device=self.device) * self.new_track_weight
        return t.cat([combined, new_track], dim=-1)

    def terminate(self) -> None:
        """Terminate the DGS module and delete the torch modules."""
        del self.sim_mods
        del self.combine
