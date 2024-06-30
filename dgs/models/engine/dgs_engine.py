"""
Engine for a full model of the dynamically gated similarity tracker.
"""

import os.path
import time

import torch as t
from lapsolver import solve_dense
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine.engine import EngineModule
from dgs.models.submission import get_submission
from dgs.models.submission.submission import SubmissionFile
from dgs.utils.config import DEF_VAL, get_sub_config
from dgs.utils.state import collate_states, EMPTY_STATE, State
from dgs.utils.torchtools import close_all_layers
from dgs.utils.track import Tracks
from dgs.utils.types import Config, Validations
from dgs.utils.utils import torch_to_numpy

dgs_eng_test_validations: Validations = {
    "submission_key": [("any", ["str", ("all", [list, ("forall", str)])])],
    # optional
    "inactivity_threshold": ["optional", int, ("gt", 0)],
    "max_track_length": ["optional", int],
    "save_images": ["optional", bool],
    "show_keypoints": ["optional", bool],
    "show_skeleton": ["optional", bool],
    "draw_kwargs": ["optional", dict],
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

    submission_key (Union[str, NodePath]):
        The key or the path of keys in the configuration containing the information about the submission file.

    Optional Test Params
    --------------------

    draw_kwargs (dict[str, any]):
        Additional keyword arguments to pass to State.draw().
        Default ``DEF_VAL.engine.dgs.draw_kwargs``.

    inactivity_threshold (int):
        The number of steps after which an inactive :class:`Track` will be removed.
        Removed tracks can be reactivated using :meth:`.Tracks.reactivate_track`.
        Use `None` to disable the removing of inactive tracks.
        Default ``DEF_VAL.tracks.inactivity_threshold``.

    max_track_length (int):
        The maximum number of :class:`.State` objects per :class:`Track`.
        Default ``DEF_VAL.track.N``.

    save_images (bool):
        Whether to save the generated image-results.
        Default ``DEF_VAL.engine.dgs.save_images``.

    show_keypoints (bool):
        Whether to show the key-point-coordinates when generating the image-results.
        Therefore, this will only have an influence, if `save_images` is `True`.
        To be drawn correctly, the detections- :class:`State` has to contain the global key-point-coordinates as
        'keypoints' and possibly the joint-visibility as 'joint_weight'.
        Default ``DEF_VAL.engine.dgs.show_skeleton``.

    show_skeleton (bool):
        Whether to connect the drawn key-point-coordinates with the human skeleton.
        This will only have an influence, if `save_images` is `True` and `show_keypoints` is `True` as well.
        To be drawn correctly, the detections- :class:`State` has to contain a valid 'skeleton_name' key.
        Default ``DEF_VAL.engine.dgs.show_skeleton``.

    """

    # The heart of the project might get a little larger...
    # pylint: disable=too-many-arguments,too-many-locals

    model: DGSModule
    tracks: Tracks
    submission: SubmissionFile

    def __init__(
        self, config: Config, model: nn.Module, test_loader: TorchDataLoader, train_loader: TorchDataLoader = None
    ):
        if not isinstance(model, DGSModule):
            raise ValueError(f"The 'model' is expected to be an instance of a DGSModule, but got '{type(model)}'.")
        super().__init__(config=config, model=model, test_loader=test_loader, train_loader=train_loader)

        self.save_images: bool = self.params_test.get("save_images", DEF_VAL["engine"]["dgs"]["save_images"])

        self.tracks = Tracks(
            N=self.params_test.get("max_track_length", DEF_VAL["track"]["N"]),
            thresh=self.params_test.get("inactivity_threshold", DEF_VAL["tracks"]["inactivity_threshold"]),
        )

        self.submission = get_submission(
            get_sub_config(config=self.config, path=self.params_test.get("submission"))["module_name"]
        )(config=self.config, path=self.params_test.get("submission"))

    def get_data(self, ds: State) -> any:
        return ds

    def get_target(self, ds: State) -> any:
        return ds["class_id"].long()

    def _track_step(self, detections: State) -> dict[str, float]:
        """Run one step of tracking."""
        N: int = len(detections)
        T: int = len(self.tracks)
        updated_tracks: dict[int, State] = {}
        new_states: list[State] = []
        batch_times: dict[str, float] = {}

        time_batch_start: float = time.time()

        # Get the current state from the Tracks and use it to compute the similarity to the current detections.
        track_state: list[State] = self.tracks.get_states()

        for ts in track_state:
            ts.clean(keys="image")
            ts.load_image_crop(store=True)

        batch_times["data"] = time.time() - time_batch_start

        if len(track_state) == 0 and N > 0:
            # No Tracks yet - every detection will be a new track!
            # Make sure to compute the embeddings for every detection, to ensure correct behavior of collate later on
            time_sim_start = time.time()
            self.model.forward(ds=detections, target=detections)
            batch_times["similarity"] = time.time() - time_sim_start
            # There are no tracks yet, therefore every detection is a new state!
            time_match_start = time.time()
            new_states += detections.split()
            batch_times["match"] = time.time() - time_match_start
        elif N > 0:
            time_sim_start = time.time()
            similarity = self.model.forward(ds=detections, target=collate_states(track_state))
            batch_times["similarity"] = time.time() - time_sim_start

            # Solve Linear sum Assignment Problem (LAP) using py-lapsolver.
            # The goal is to obtain the best combination of Track-IDs and detection-IDs given the probabilities of
            # a similarity-matrix with a shape of [N x T].
            # Because the algorithm returns the lowest scores, we need to compute 1-sim as cost matrix.
            # The result is a list of N 2-tuples containing the position
            time_match_start = time.time()
            cost_matrix = t.ones_like(similarity) - similarity
            # lapsolver uses numpy arrays instead of torch, therefore, convert but loose computational graph
            cost_matrix = torch_to_numpy(cost_matrix)
            rids, cids = solve_dense(cost_matrix)  # rids and cids are ndarray of shape [N]

            assert (
                N == len(rids) == len(cids)
            ), f"expected shapes to match - N: {N}, states: {len(track_state)}, rids: {len(rids)}, cids: {len(cids)}"

            states: list[State] = detections.split()
            tids = self.tracks.ids
            for rid, cid in zip(rids, cids):
                if cid < T and cid in tids:
                    updated_tracks[cid] = states[rid]
                else:
                    new_states.append(states[rid])
            batch_times["match"] = time.time() - time_match_start

        # update tracks
        time_track_update_start = time.time()
        self.tracks.add(tracks=updated_tracks, new=new_states)

        batch_times["track"] = time.time() - time_track_update_start
        batch_times["batch"] = time.time() - time_batch_start
        if N > 0:
            batch_times["indiv"] = batch_times["batch"] / N

        return batch_times

    @t.no_grad()
    def test(self) -> dict[str, any]:
        """Test the DGS Tracker"""
        # pylint: disable=too-many-statements

        results: dict[str, any] = {}
        detections: list[State]
        detection: State
        frame_idx: int = 0

        # set model to evaluation mode and freeze / close all layers
        self.set_model_mode("eval")
        close_all_layers(self.model)

        self.logger.debug(f"#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####")
        self.logger.debug("Loading, extracting, and predicting data, this might take a while...")

        for detections in tqdm(self.test_dl, desc="DataLoader", leave=False):
            for detection in detections:

                N: int = len(detections)

                batch_times = self._track_step(detections=detection)

                # get active states and skip adding if there are no active states
                active_list = self.tracks.get_active_states()

                # handle empty active list, by setting the filepath, image_id, and frame_id
                if len(active_list) == 0 or all(a.B == 0 for a in active_list):
                    active = EMPTY_STATE.copy()
                    active.filepath = detection.filepath
                    active["image_id"] = detection["image_id"]
                    active["frame_id"] = detection["frame_id"]
                else:
                    active = collate_states(active_list)

                # store current submission data
                self.submission.append(active)

                # print debug info
                # Add timings and other metrics to the writer
                self.writer.add_scalar(tag="Test/BatchSize", scalar_value=N, global_step=frame_idx)
                self.writer.add_scalars(
                    main_tag="Test/time",
                    tag_scalar_dict={**batch_times},
                    global_step=frame_idx,
                )
                # print the resulting image if requested
                if self.save_images:
                    active.draw(
                        save_path=os.path.join(self.log_dir, f"./images/{frame_idx:05d}.png"),
                        show_kp=(
                            self.params_test.get("show_keypoints", DEF_VAL["engine"]["dgs"]["show_keypoints"])
                            if detection.B > 0
                            else False
                        ),
                        show_skeleton=(
                            self.params_test.get("show_skeleton", DEF_VAL["engine"]["dgs"]["show_skeleton"])
                            if detection.B > 0
                            else False
                        ),
                        **self.params_test.get("draw_kwargs", DEF_VAL["engine"]["dgs"]["draw_kwargs"]),
                    )

                    frame_idx += 1

            # free up memory by removing the images and crops
            for d in detections:
                d.clean()

        self.submission.save()

        self.logger.debug(f"#### Finished Evaluating {self.name} ####")

        return results

    @t.no_grad()
    def predict(self) -> None:
        """Given test data, predict the results without evaluation."""
        # set model to evaluation mode and freeze / close all layers
        self.set_model_mode("eval")
        close_all_layers(self.model)
        frame_idx: int = 0

        self.logger.info(f"#### Start Prediction {self.name} ####")
        self.logger.info("Loading, extracting, and predicting data, this might take a while...")
        detections: list[State]
        # batch get data from the data loader
        for detections in tqdm(self.test_dl, desc="DataLoader"):
            for detection in tqdm(detections, desc="Tracker", leave=False):
                _ = self._track_step(detections=detection)

                active = collate_states(self.tracks.get_active_states())

                # store current submission data
                self.submission.append(active)

                out_fp = os.path.join(self.log_dir, f"./images/{frame_idx:05d}.png")
                if detection.B > 0:
                    active = collate_states(self.tracks.get_active_states())
                    active.draw(
                        save_path=out_fp,
                        show_kp=self.params_test.get("show_keypoints", DEF_VAL["engine"]["dgs"]["show_keypoints"]),
                        show_skeleton=self.params_test.get("show_skeleton", DEF_VAL["engine"]["dgs"]["show_skeleton"]),
                        **self.params_test.get("draw_kwargs", DEF_VAL["engine"]["dgs"]["draw_kwargs"]),
                    )
                else:
                    detection.draw(
                        save_path=out_fp,
                        show_kp=False,
                        show_skeleton=False,
                        **self.params_test.get("draw_kwargs", DEF_VAL["engine"]["dgs"]["draw_kwargs"]),
                    )

                for track in self.tracks.values():
                    track[-1].clean()
                # move to the next frame
                frame_idx += 1

        self.submission.save()

        self.logger.info(f"#### Finished Prediction {self.name} ####")

    def _get_train_loss(self, data: State, _curr_iter: int) -> t.Tensor:
        raise NotImplementedError

    def terminate(self) -> None:
        self.submission.terminate()
        self.model.terminate()
        del self.tracks
        super().terminate()
