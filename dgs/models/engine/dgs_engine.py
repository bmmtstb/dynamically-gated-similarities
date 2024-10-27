"""
Engine for a full model of the dynamically gated similarity tracker.
"""

import os.path
import time
from datetime import timedelta

import torch as t
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader as TDataLoader
from tqdm import tqdm

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine.engine import EngineModule
from dgs.models.metric.metric import compute_near_k_accuracy
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.submission import get_submission
from dgs.models.submission.submission import SubmissionFile
from dgs.utils.config import DEF_VAL, get_sub_config
from dgs.utils.state import collate_states, EMPTY_STATE, State
from dgs.utils.timer import DifferenceTimers
from dgs.utils.torchtools import load_checkpoint
from dgs.utils.track import Tracks
from dgs.utils.types import Config, FilePath, NodePath, Results, Validations
from dgs.utils.utils import torch_to_numpy

dgs_eng_test_validations: Validations = {
    "submission": ["NodePath"],
    # optional
    "inactivity_threshold": ["optional", int, ("gt", 0)],
    "max_track_length": ["optional", int],
    "save_images": ["optional", bool],
    "show_keypoints": ["optional", bool],
    "show_skeleton": ["optional", bool],
    "draw_kwargs": ["optional", dict],
}

dgs_eng_train_validations: Validations = {
    # optional
    "submission": ["optional", "NodePath"],
    "acc_k_train": ["optional", list, ("forall", ["number"])],
    "acc_k_eval": ["optional", list, ("forall", ["number"])],
    "eval_accuracy": ["optional", bool],
}


class DGSEngine(EngineModule):
    """An engine class for training and testing the dynamically gated similarity tracker with static or dynamic gates.

    For this model:

    * ``get_data()`` should return the same as this similarity functions :meth:`SimilarityModule.get_data` call
    * ``get_target()`` should return the class IDs of the :class:`State` object
    * ``train_dl`` contains the training data as a torch DataLoader containing a :class:`ImageHistoryDataset` dataset.
      Additionally, the training data should contain all the training sequences and not just a single video.
    * ``test_dl`` contains the test data as a torch DataLoader
      containing a regular :class:`ImageDataset` or class:`VideoDataset` datasets
    * ``val_dl`` contains the validation data.
      The validation data can be one of the following,
      depending on the configuration of ``params_train["eval_accuracy"]``:

      * If ``eval_accuracy`` is ``True``,
        the evaluation data is as a torch DataLoader containing a :class:`ImageHistoryDataset` dataset.
        Additionally, the validation data should contain all the validation sequences and not just a single video.
      * If ``eval_accuracy`` is ``False``, the evaluation data is as a torch DataLoader
        containing a regular :class:`ImageDataset` or class:`VideoDataset` datasets.
        With one dataset per video.


    Train Params
    ------------

    Test Params
    -----------

    submission (Union[str, NodePath]):
        The key or the path of keys in the configuration containing the information about the submission file,
        which is used to save the test data.

    Optional Train Params
    ---------------------

    acc_k_train (list[int|float], optional):
        A list of values used during training to check whether the accuracy lies within a margin of k percent.
        Default ``DEF_VAL.engine.dgs.acc_k_train``.

    acc_k_eval (list[int|float], optional):
        A list of values used during evaluation to check whether the accuracy lies within a margin of k percent.
        Default ``DEF_VAL.engine.dgs.acc_k_eval``.

    eval_accuracy (bool, optional):
        Whether to evaluate the alpha-prediction accuracy or the |MOTA| / |HOTA| of the model during evaluation.
        Default ``DEF_VAL.engine.dgs.eval_accuracy``.

    submission (Union[str, NodePath]):
        The key or the path of keys in the configuration containing the information about the submission file,
        which is used to save the evaluation data, if ``eval_accuracy`` is ``False``.


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
    """The DGS module containing the similarity models and the alpha model."""

    tracks: Tracks
    """The tracks object containing all the active tracks of this engine."""

    submission: SubmissionFile
    """The submission file to store the results when running the tests."""

    val_dl: TDataLoader
    """The torch DataLoader containing the validation data."""

    train_dl: TDataLoader
    """The torch DataLoader containing the train data."""

    def __init__(
        self,
        config: Config,
        path: NodePath,
        model: nn.Module,
        *,
        test_loader: TDataLoader = None,
        val_loader: TDataLoader = None,
        train_loader: TDataLoader = None,
        **_kwargs,
    ):
        if not isinstance(model, DGSModule):
            raise ValueError(f"The 'model' is expected to be an instance of a DGSModule, but got '{type(model)}'.")
        super().__init__(
            config=config,
            path=path,
            model=model,
            test_loader=test_loader,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        # TEST - get params from config
        self.validate_params(dgs_eng_test_validations, "params_test")
        self.save_images: bool = self.params_test.get("save_images", DEF_VAL["engine"]["dgs"]["save_images"])

        # TRAIN - get params from config
        if self.is_training:
            self.validate_params(dgs_eng_train_validations, "params_train")

        # initialize the tracks
        self.tracks = Tracks(
            N=self.params_test.get("max_track_length", DEF_VAL["track"]["N"]),
            thresh=self.params_test.get("inactivity_threshold", DEF_VAL["tracks"]["inactivity_threshold"]),
        )

    def get_data(self, ds: State) -> list[t.Tensor]:
        """Use the similarity models of the DGS module to obtain the similarity data of the current detections.

        For the similarity engine, the data consists of a list of all the input data for the similarities.
        This means, that for the visual similarity, the embedding is returned,
        and for the IoU or OKS similarities, the bbox and key point data is returned.
        The :func:`get_data` function will be called twice, once for the current time-step and once for the previous.
        """
        return [sm.get_train_data(ds) for sm in self.model.sim_mods]

    def get_target(self, ds: State) -> t.Tensor:
        """Get the target data.

        For the similarity engine, the target data consists of the dataset-unique class-id.
        The :func:`get_target` function will be called twice, once for the current time-step and once for the previous.
        """
        try:
            return ds.class_id
        except KeyError as e:
            raise KeyError(f"State: {ds.data}") from e

    @enable_keyboard_interrupt
    def _track_step(self, detections: State, frame_idx: int, name: str, timers: DifferenceTimers) -> None:
        """Run one step of tracking.

        Args:
            detections: The detections for the current frame as a single :class:`State` object.
            frame_idx: The current frame index used for logger stuff.
            name: The name of the current step used for logger stuff.
            timers: The timers to store the timing information of every step.
        """
        N: int = len(detections)
        updated_tracks: dict[int, State] = {}
        new_states: list[State] = []

        time_batch_start: float = time.time()

        # Get the current state from the Tracks and use it to compute the similarity to the current detections.
        track_states, tids = self.tracks.get_states()

        for ts in track_states:
            ts.load_image_crop(store=True)
            ts.clean(keys=["image"])

        timers.add(name="data", prev_time=time_batch_start)

        if len(track_states) == 0 and N > 0:
            # No Tracks yet - every detection will be a new track!
            # Make sure to compute the embeddings for every detection, to ensure correct behavior of collate later on
            time_sim_start = time.time()
            _ = self.model.forward(ds=detections, target=detections, alpha_inputs=self.get_data(detections))
            timers.add(name="similarity", prev_time=time_sim_start)
            # There are no tracks yet, therefore every detection is a new state!
            time_match_start = time.time()
            new_states += detections.split()
            timers.add(name="match", prev_time=time_match_start)
        elif N > 0:
            time_sim_start = time.time()
            similarity = self.model.forward(
                ds=detections, target=collate_states(track_states), alpha_inputs=self.get_data(detections)
            )
            timers.add(name="similarity", prev_time=time_sim_start)

            # Solve Linear sum Assignment Problem (LAP/LSA).
            # Goal: obtain the best combination of Track-IDs and detection-IDs given the probabilities in the
            # similarity-matrix. Due to adding zeros for empty tracks, the SM has a shape of [N x (T+N)].
            # The LSA always returns indices of length N because N <= T+N for all positive T.
            # The result is a list of N 2-tuples containing the position
            time_match_start = time.time()
            # scipy uses numpy arrays instead of torch, therefore, convert -> but loose computational graph
            sim_matrix = torch_to_numpy(similarity)
            rids, cids = linear_sum_assignment(sim_matrix, maximize=True)  # rids and cids are ndarray of shape [N]

            assert 0 <= (cost := sim_matrix[rids, cids].sum()) <= N, (
                f"expected the cost matrix to be between 0 and N, "
                f"got r: {rids}, c: {cids}, cm: {sim_matrix}, N: {N}, cost: {cost}"
            )
            self.writer.add_scalar(tag=f"{name}/cost", scalar_value=cost, global_step=frame_idx)

            assert (
                N == len(rids) == len(cids)
            ), f"expected shapes to match - N: {N}, states: {len(track_states)}, rids: {len(rids)}, cids: {len(cids)}"

            states: list[State] = detections.split()
            for rid, cid in zip(rids, cids):
                if cid < len(tids):
                    updated_tracks[tids[cid]] = states[rid]
                else:
                    new_states.append(states[rid])
            timers.add(name="match", prev_time=time_match_start)

        # update tracks
        time_track_update_start = time.time()
        self.tracks.add(tracks=updated_tracks, new=new_states)
        timers.add(name="track", prev_time=time_track_update_start)

        # get the overall timing of the batch
        batch_time = timers.add(name="batch", prev_time=time_batch_start)
        timers.add(name="indiv", prev_time=0.0, now=batch_time / N if N > 0 else batch_time)

    @t.no_grad()
    def test(self) -> Results:
        """Test the DGS Tracker on the test_dl."""

        if self.test_dl is None:
            raise ValueError("The test data loader is required for testing.")

        # set model to evaluation mode and freeze / close all layers
        self.set_model_mode("eval")

        # set up submission data
        self.submission = get_submission(
            get_sub_config(config=self.config, path=self.params_test.get("submission"))["module_name"]
        )(config=self.config, path=self.params_test.get("submission"))

        self.logger.debug(f"#### Start Test {self.name} - Epoch {self.curr_epoch} ####")
        start_time: float = time.time()

        self._track(dl=self.test_dl, name="Test")

        self.logger.debug(
            f"#### Finished Test {self.name} Epoch {self.curr_epoch} "
            f"in {str(timedelta(seconds=round(time.time() - start_time)))} ####"
        )

        return {}

    @t.no_grad()
    def predict(self) -> None:
        """Given test data, predict the results without evaluation."""
        if self.test_dl is None:
            raise ValueError("The test data loader is required for testing.")

        # set model to evaluation mode and freeze / close all layers
        self.set_model_mode("eval")

        frame_idx: int = int(self.curr_epoch * len(self.test_dl) * self.test_dl.batch_size)

        self.submission = get_submission(
            get_sub_config(config=self.config, path=self.params_test.get("submission"))["module_name"]
        )(config=self.config, path=self.params_test.get("submission"))

        self.logger.info(f"#### Start Prediction {self.name} ####")
        self.logger.info("Loading, extracting, and predicting data, this might take a while...")

        start_time: float = time.time()
        timers: DifferenceTimers = DifferenceTimers()
        detections: list[State]

        # batch get data from the data loader
        for detections in tqdm(self.test_dl, desc="Predict"):
            for detection in tqdm(detections, desc="Tracker", leave=False):
                self._track_step(detections=detection, frame_idx=frame_idx, name="Predict", timers=timers)

                active = collate_states(self.tracks.get_active_states())

                # store current submission data
                self.submission.append(active)

                out_fp = os.path.join(self.log_dir, f"./images/{frame_idx:05d}.png")
                if detection.B > 0:
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
        self.tracks.reset()

        self.logger.info(
            f"#### Finished Prediction {self.name} in {str(timedelta(seconds=round(time.time() - start_time)))} ####"
        )

    def _get_accuracy(self, data_old: State, data_new: State) -> t.Tensor:
        """Given the old and new state, compute the accuracy of the similarities as the number of correct matches
        divided by the total number of detections.
        """
        with t.no_grad():
            old_ids = self.get_target(data_old).flatten()  # [T]
            new_ids = self.get_target(data_new).flatten()  # [D]

            # concat all IDs from new_ids, which are not present in old_ids, to the old_ids
            # only add new_ids that are not in old_ids
            if len(old_ids) > 0 and len(new_ids) > 0:
                combined_ids = t.cat(
                    [old_ids, new_ids[~(new_ids.reshape((-1, 1)) == old_ids.reshape((1, -1))).max(dim=1)[0]]]
                )
            else:
                # return early with accuracy of zero, because there are no new ids and or no old ids
                return t.zeros(len(self.model.sim_mods), device=self.device, dtype=t.float32)

            # Obtain a matrix describing the matches between the new_ids and the old_ids.
            # if there is no match, use the ids of newly created tracks (the second  part of the combined_ids)
            # With B>1 there might be multiple ID matches, all of those will be tested / counted later on.
            # [D x D+T]
            all_matches = new_ids.reshape(-1, 1) == combined_ids.reshape(1, -1)

            # get the similarity matrix as [S x D x (T + D)]
            similarities = t.stack([m(data_new, data_old) for m in self.model.sim_mods])
            # get the positions of the maximum similarity for each detection
            max_sim_indices = t.argmax(similarities, dim=-1)  # [S x D]
            # The number of correct matches can be obtained by counting the indices where the similarity is highest
            # and the target-ID of the prediction is the same as the target-ID of the ground truth.
            # [S]
            # fixme: is it possible to do this without leaving pytorch or adding python loops?
            nof_correct = t.stack([all_matches[t.arange(len(new_ids)), msi].count_nonzero() for msi in max_sim_indices])
            # the training target is the ratio of correct matches to the total number of detections
            # making sure to handle the cases where either no detections or no correct matches are present
            accuracy = (nof_correct / len(data_new)).nan_to_num(nan=0.0, posinf=0.0)  # [S]
            assert t.all(
                t.bitwise_or(accuracy <= 1.0, accuracy >= 0.0)
            ), f"expected accuracy to lie within [0,1], got: {accuracy}"
            return accuracy

    @enable_keyboard_interrupt
    def _get_train_loss(self, data: list[State], _curr_iter: int) -> t.Tensor:
        """Calculate the loss for the current frame."""
        # fixme: this should support data_old with arbitrary size (or more precise an ImageHistoryDataset with L != 1)

        assert isinstance(data, list) and len(data) == 2, f"Data must be a list of length 2. but got {len(data)}"
        data_old, data_new = data

        target = self._get_accuracy(data_old=data_old, data_new=data_new)  # [S]

        with t.enable_grad():
            # get the input data of the similarity modules for the current step
            curr_sim_data = self.get_data(data_new)  # [D]

            # For each of the similarity modules, compute the alpha value of each of the respective inputs.
            # [S x D]
            alpha = t.stack([a_m(curr_sim_data[i]).flatten() for i, a_m in enumerate(self.model.combine.alpha_models)])
            # make sure the target and input have the same shape, by repeating the target for each input
            loss = self.loss(alpha, target.expand(1, len(data_new)))
        return loss

    def _track(self, dl: TDataLoader, name: str) -> None:
        """Track the data in the DataLoader."""
        frame_idx: int = int(self.curr_epoch * len(dl) * dl.batch_size)

        # reset submission and track data before starting the tracking
        self.submission.clear()
        self.tracks.reset()

        timers = DifferenceTimers()

        for detections in tqdm(dl, desc=f"Track-{name}-ep{self.curr_epoch}", leave=False):
            for detection in detections:

                N: int = len(detections)

                self._track_step(detections=detection, frame_idx=frame_idx, name="Track", timers=timers)

                # get active states and skip adding if there are no active states
                active_list = self.tracks.get_active_states()

                # handle empty active list, by setting the filepath, image_id, and frame_id
                if len(active_list) == 0 or all(a.B == 0 for a in active_list):
                    active = EMPTY_STATE.copy()
                    active.filepath = detection.filepath
                    if "image_id" in detection:
                        active.data["image_id"] = detection["image_id"]
                    if "frame_id" in detection:
                        active.data["frame_id"] = detection["frame_id"]
                    active.data["pred_tid"] = t.empty(0, dtype=t.long, device=detection.device)
                else:
                    active = collate_states(active_list)

                # store current submission data
                self.submission.append(active)

                # ############ #
                # END OF FRAME #
                # ############ #

                # print the resulting image if requested
                if self.save_images:
                    time_drawing = time.time()
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
                    timers.add(name="draw", prev_time=time_drawing)

                # Add timings and other metrics to the writer
                self.writer.add_scalar(tag=f"{name}/Track/batch_size", scalar_value=N, global_step=frame_idx)
                self.writer.add_scalars(
                    main_tag=f"{name}/Track/time", tag_scalar_dict=timers.get_last(), global_step=frame_idx
                )

                # remove unused images and crops
                active.clean()

                frame_idx += 1

            # ############ #
            # END OF BATCH #
            # ############ #

            # free up memory by removing the images and crops
            for d in detections:
                d.clean()

        # ############### #
        # END OF Tracking #
        # ############### #

        self.writer.add_hparams(
            hparam_dict={
                "name": name,
                "batch_size": dl.batch_size,
                "len_dl": len(dl),
                "curr_epoch": self.curr_epoch,
                "inactivity_threshold": self.tracks.inactivity_threshold,
                "max_track_length": self.tracks.N,
            },
            metric_dict=timers.get_avgs(),
            global_step=self.curr_epoch,
            run_name=f"Track-{name}-{self.name_safe}",
        )

        self.submission.save()
        self.submission.clear()
        self.tracks.reset()

    @t.no_grad()
    def _eval_alpha(self) -> Results:
        """Evaluate the alpha model by computing the accuracy of the alpha prediction."""
        frame_idx: int = self.curr_epoch * len(self.val_dl) * self.val_dl.batch_size
        ks = self.params_train.get("acc_k_eval", DEF_VAL["engine"]["dgs"]["acc_k_eval"])
        S = len(self.model.sim_mods)

        # there might be multiple similarities, therefore create a base value for every k for every similarity
        results: dict[str | int, any] = {
            "N": 0,
            **dict(
                zip([f"{sim_mod.module_name}-{k}" for k in ks for sim_mod in self.model.sim_mods], [0] * len(ks) * S)
            ),
        }
        for data in tqdm(self.val_dl, desc=f"Evaluate - epoch {self.curr_epoch}", leave=False):

            assert isinstance(data, list) and len(data) == 2, "Data must be a list of length 2."
            data_old, data_new = data

            D = len(data_new)
            # nothing to do, if there are no detections
            if D > 0:
                accuracy = self._get_accuracy(data_old=data_old, data_new=data_new)  # [S]

                # get the input data of the similarity modules for the current step
                # and use those to compute all the alpha scores [S]
                curr_sim_data = self.get_data(data_new)  # [D]
                alpha: t.Tensor = t.stack(
                    [a_m(curr_sim_data[i]).flatten() for i, a_m in enumerate(self.model.combine.alpha_models)]
                )  # [S x D]

                # compare alpha against the correct similarities
                # alpha and the accuracy are of shape [S (x D)], therefore evaluate for every s in S
                for s_i, alpha_i in enumerate(alpha):
                    sim_name: str = self.model.sim_mods[s_i].module_name
                    accuracies = compute_near_k_accuracy(a_pred=alpha_i, a_targ=accuracy.repeat(D), ks=ks)
                    for k, acc in accuracies.items():
                        results[f"{sim_name}-{k}"] += round(float(acc) * float(D))
                    self.writer.add_scalars(
                        main_tag="Eval/accu",
                        tag_scalar_dict={f"{sim_name}-{k}": v for k, v in accuracies.items()},
                        global_step=frame_idx,
                    )

                results["N"] += D

            # clean up data to save memory
            if isinstance(data, State):
                data.clean()
            elif isinstance(data, (list, tuple)):
                for d in data:
                    d.clean()

            # End of frame
            frame_idx += 1

        # compute overall accuracy of this dataset given partially data (across all the similarities of this module)
        # results will be written to the writer in the evaluate method
        for k in ks:
            k_name: str = f"acc-{k:05.1f}"  # format 51.4% as 051.4
            results[k_name] = sum(float(results[f"{sim_mod.module_name}-{k}"]) for sim_mod in self.model.sim_mods) / (
                float(results["N"]) * S
            )
            # save per-similarity accuracies as percentages
            for sim_mod in self.model.sim_mods:
                results.pop(f"{sim_mod.module_name}-{k}")

        return results

    def _eval_tracking(self) -> None:
        """Prepare to evaluate the tracking performance similar to test but on the evaluation data."""
        if "submission" not in self.params_train:
            raise ValueError("The 'submission' key is required in the 'train' parameters  if 'eval_accuracy' is False.")
        self.submission = get_submission(
            get_sub_config(config=self.config, path=self.params_train.get("submission"))["module_name"]
        )(config=self.config, path=self.params_train.get("submission"))

        self._track(dl=self.val_dl, name="Eval")

    @t.no_grad()
    def evaluate(self) -> Results:
        r"""Run the model evaluation on the eval data.

        Test whether the predicted alpha probability (:math:`\alpha_{\mathrm{pred}}`)
        matches the number of correct predictions (:math:`\alpha_{\mathrm{correct}}`)
        divided by the total number of predictions (:math:`N`).

        With :math:`\alpha{\mathrm{pred}} = \frac{\alpha_{\mathrm{correct}}}{N}`
        :math`\alpha{\mathrm{pred}}` is counted as correct if
        :math:`\alpha{\mathrm{pred}}-k \leq \alpha{\mathrm{correct}} \leq \alpha{\mathrm{pred}}+k`.
        """
        self.logger.debug("Start Evaluation")

        self.set_model_mode("eval")

        start_time: float = time.time()

        if self.params_train.get("eval_accuracy", DEF_VAL["engine"]["dgs"]["eval_accuracy"]):
            results = self._eval_alpha()
            self.print_results(results)
            self.write_results(results, prepend="Eval")
        else:
            self._eval_tracking()
            results = {}

        self.logger.debug(
            f"#### Evaluation of {self.name} Epoch {self.curr_epoch} "
            f"complete in {str(timedelta(seconds=round(time.time() - start_time)))} ####"
        )
        return results

    def load_combine_alpha_weights(self, fp: FilePath, new_id: int = 0, old_id: int = 0) -> None:
        """Given the path to a file containing at least the data of one module checkpoint, load the weights of
        the ``combine.alpha_weights`` module.

        Notes:
            During training the DGSEngine was trained with a single alpha model.
            For testing or (non accuracy) evaluation, multiple alpha values are required.
            Therefore, the ``combine.alpha_models`` now contains more than one AlphaGenerator instance.
            Thus, the indices of the state dict have to be modified accordingly.

            Additionally, in case of the visual embedding generation modules, there are more parameters saved in the
            checkpoint file, which should not be loaded by this function.

        Args:
            fp: The path to the checkpoint file
            new_id: The ID at which index of the alpha weight modules to insert the loaded weights.
            old_id: The old ID. Necessary only if there are multiple ``combine.alpha_models`` s in a single checkpoint.
                E.g. when multiple alpha weight generators have been trained in unison.
        """
        checkpoint_data = load_checkpoint(fpath=fp)
        state_dict = checkpoint_data["model"] if "model" in checkpoint_data else checkpoint_data

        # Only load combine.alpha_models and ignore everything else. (e.g. visual embed gen models)
        new_state_dict = {
            str(k).replace(f"combine.alpha_models.{int(old_id)}.", f"combine.alpha_models.{int(new_id)}."): v
            for k, v in state_dict.items()
            if k.startswith("combine.alpha_models")
        }

        _, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if len(unexpected) != 0:
            raise ValueError(f"got unexpected keys: {unexpected}")

    def terminate(self) -> None:
        if hasattr(self, "submission"):
            self.submission.clear()
            self.submission.terminate()
        if hasattr(self, "model"):
            self.model.terminate()
            del self.model
        if hasattr(self, "tracks"):
            self.tracks.reset()
            del self.tracks
        super().terminate()
