"""
Engine for a full model of the dynamically gated similarity tracker.
"""

import time

import torch
from lapsolver import solve_dense
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine.engine import EngineModule
from dgs.utils.config import DEF_CONF
from dgs.utils.state import State
from dgs.utils.timer import DifferenceTimer
from dgs.utils.torchtools import close_all_layers
from dgs.utils.track import Track, Tracks
from dgs.utils.types import Config, Validations
from dgs.utils.utils import torch_to_numpy

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

        self.max_track_len: int = self.params_test.get("max_track_length", DEF_CONF.track.N)
        self.tracks = Tracks(thresh=self.params_test.get("inactivity_threshold", DEF_CONF.tracks.inactivity_threshold))

    def get_data(self, ds: State) -> any:
        return ds

    def get_target(self, ds: State) -> any:
        return ds["class_id"].long()

    def test(self) -> dict[str, any]:
        """Test the DGS Tracker"""
        # pylint: disable=too-many-statements

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
        track_t: DifferenceTimer = DifferenceTimer()

        time_batch_start: float = time.time()

        for batch_idx, detections in enumerate(self.test_dl):
            # fixme reset tracks at the end of every sub-dataset
            N: int = len(detections)
            T: int = len(self.tracks)
            batch_times: dict[str, float] = {}

            updated_tracks: dict[int, State] = {}
            new_tracks: list[Track] = []

            # Get the current state from the Tracks and use it to compute the similarity to the current detections.
            track_state: State = self.tracks.get_states()
            data_t.add(time_batch_start)
            batch_times["data"] = data_t[-1]

            if len(track_state) == 0 and N > 0:
                # No Tracks yet - every detection will be a new track!
                time_match_start = time.time()
                states: list[State] = detections.split()
                for state in states:
                    t = Track(N=self.max_track_len, states=[state])
                    new_tracks.append(t)
                match_t.add(time_match_start)
                batch_times["match"] = match_t[-1]
            elif N > 0:
                time_sim_start = time.time()
                similarity = self.model.forward(ds=detections, target=track_state)
                similarity_t.add(time_sim_start)
                batch_times["similarity"] = similarity_t[-1]

                # Solve Linear sum Assignment Problem (LAP) using py-lapsolver.
                # The goal is to obtain the best combination of Track-IDs and detection-IDs given the probabilities of
                # a similarity-matrix with a shape of [N x T].
                # Because the algorithm returns the lowest scores, we need to compute 1-sim as cost matrix.
                # The result is a list of N 2-tuples containing the position
                time_match_start = time.time()
                cost_matrix = torch.ones_like(similarity) - similarity
                # lapsolver uses numpy arrays instead of torch, therefore, convert but loose computational graph
                cost_matrix = torch_to_numpy(cost_matrix)
                rids, cids = solve_dense(cost_matrix)  # rids and cids are ndarray of shape [N]

                states: list[State] = detections.split()
                assert len(states) == len(rids) == len(cids), "expected shapes to match"

                for rid, cid in zip(rids, cids):
                    if cid < T:
                        updated_tracks[cid] = states[rid]
                    else:
                        t = Track(N=self.max_track_len, states=[states[rid]])
                        new_tracks.append(t)
                match_t.add(time_match_start)
                batch_times["matching"] = match_t[-1]

            # update tracks
            time_track_update_start = time.time()
            self.tracks.add(tracks=updated_tracks, new_tracks=new_tracks)
            track_t.add(time_track_update_start)
            batch_times["track"] = track_t[-1]

            batch_t.add(time_batch_start)
            batch_times["batch"] = batch_t[-1]
            if N > 0:
                batch_times["indiv"] = batch_t[-1] / N
            self.writer.add_scalar(tag="Test/BatchSize", scalar_value=N, global_step=batch_idx)

            self.writer.add_scalars(
                main_tag="Test/time",
                tag_scalar_dict={**batch_times},
                global_step=batch_idx,
            )
            # reset timer for next batch
            time_batch_start = time.time()

        return results

    def _get_train_loss(self, data: State, _curr_iter: int) -> torch.Tensor:
        raise NotImplementedError
