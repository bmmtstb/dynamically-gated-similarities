"""
Engine for a full model of the dynamically gated similarity tracker.
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine import EngineModule
from dgs.utils.config import DEF_CONF
from dgs.utils.state import State
from dgs.utils.timer import DifferenceTimer
from dgs.utils.torchtools import close_all_layers
from dgs.utils.track import Tracks
from dgs.utils.types import Config, Validations

dgs_engine_validations: Validations = {
    "inactivity_threshold": ["optional", int, ("gt", 0)],
}


class DGSEngine(EngineModule):
    """An engine class for training and testing the dynamically gated similarity tracker.

    For this model:

    - ``get_data()`` should return the State
    - ``get_target()`` should return the target class IDs
    - ``train_dl`` contains the training data as usual
    - ``test_dl`` TODO

    Train Params
    ------------

    Test Params
    -----------

    inactivity_threshold (int):
        The number of steps after which an inactive Track will be forgotten.
        Default `DEF_CONF.tracks.inactivity_threshold`.

    Optional Test Params
    --------------------

    """

    # The heart of the project might get a little larger...
    # pylint: disable=too-many-arguments,too-many-locals

    model: DGSModule
    tracks: Tracks

    def __init__(
        self, config: Config, model: nn.Module, test_loader: TorchDataLoader, train_loader: TorchDataLoader = None
    ):
        if not isinstance(model, DGSModule):
            raise ValueError(f"The 'model' is expected to be an instance of a DGSModule, but got '{type(model)}'.")
        super().__init__(config=config, model=model, test_loader=test_loader, train_loader=train_loader)

        self.tracks = Tracks(thresh=self.params_test.get("inactivity_threshold", DEF_CONF.tracks.inactivity_threshold))

    def get_data(self, ds: State) -> any:
        return ds

    def get_target(self, ds: State) -> any:
        return ds["class_id"].long()

    def test(self) -> dict[str, any]:
        """Test the DGS Tracker"""
        results: dict[str, any] = {}
        detections: State

        # set model to evaluation mode and freeze / close all layers
        self.set_model_mode("eval")
        close_all_layers(self.model)

        self.logger.info(f"#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####")
        self.logger.info("Loading, extracting, and predicting data, this might take a while...")

        # set up timers
        data_t: DifferenceTimer = DifferenceTimer()
        batch_t: DifferenceTimer = DifferenceTimer()
        similarity_t: DifferenceTimer = DifferenceTimer()
        match_t: DifferenceTimer = DifferenceTimer()

        time_batch_start: float = time.time()

        for batch_idx, detections in enumerate(self.test_dl):
            N = len(detections)

            # Get the current state from the Tracks and use it to compute the similarity to the current detections.
            track_state: State = self.tracks.get_states()

            data_t.add(time_batch_start)
            time_sim_start = time.time()

            similarity = self.model.forward(ds=detections, target=track_state)

            similarity_t.add(time_sim_start)
            time_match_start = time.time()

            # munkres / hungarian matching to obtain track-id probabilities
            _ = (similarity,)

            match_t.add(time_match_start)

            # update tracks
            self.tracks.add(tracks={}, new_tracks=[])

            batch_t.add(time_batch_start)

            self.writer.add_scalars(
                main_tag="Test/time",
                tag_scalar_dict={
                    "data": data_t[-1],
                    "similarity": similarity_t[-1],
                    "matching": match_t[-1],
                    "batch": batch_t[-1],
                    "indiv": batch_t[-1] / N,
                },
                global_step=batch_idx,
            )
            self.writer.add_scalar(tag="Test/BatchSize", scalar_value=N, global_step=batch_idx)

            time_batch_start = time.time()

        return results

    def _get_train_loss(self, data: State, _curr_iter: int) -> torch.Tensor:
        raise NotImplementedError
