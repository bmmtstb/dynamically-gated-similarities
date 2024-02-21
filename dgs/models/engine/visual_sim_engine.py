"""
Engine for visual embedding training and testing.
"""

import time
import warnings
from datetime import timedelta

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from dgs.models.engine.engine import EngineModule
from dgs.models.metric import metric, METRICS
from dgs.models.module import enable_keyboard_interrupt
from dgs.utils.states import DataSample
from dgs.utils.timer import DifferenceTimer
from dgs.utils.types import Config, Validations

train_validations: Validations = {
    "nof_classes": [int, ("gt", 0)],
    # optional
    "topk_acc": ["optional", ("forall", [int, ("gt", 0)])],
}

test_validations: Validations = {
    "metric": [("any", ["callable", ("in", METRICS.keys())])],
    # optional
    "metric_kwargs": ["optional", dict],
    "topk_cmc": ["optional", ("forall", [int, ("gt", 0)])],
    "write_embeds": ["optional", ("len", 2), ("forall", bool)],
}


class VisualSimilarityEngine(EngineModule):
    """An engine class for training and testing visual similarities using visual embeddings.

    For this model:

    - ``get_data()`` should return the image crop
    - ``get_target()`` should return the target pIDs
    - ``train_dl`` contains the training data as usual
    - ``test_dl`` contains the query data
    - ``val_dl`` contains the gallery data


    Train Params
    ------------

    nof_classes (int):
        The number of classes in the training set.
    topk_acc (list[int], optional):
        The values for k for the top-k accuracy evaluation during training.
        Default [1].

    Test Params
    -----------

    metric (str|callable):
        The name or class of the metric used during testing / evaluation.
        The metric in the ``VisualSimilarityEngine`` is only used
        to compute the distance between the query and gallery embeddings.
        Therefore, a distance-based metric should be used.

        It is possible to pass additional initialization kwargs to the metric
        by adding them to the ``metric_kwargs`` parameter.


    Optional Test Params
    --------------------

    metric_kwargs (dict, optional):
        Specific kwargs for the metric.
        Default {}.
    topk_cmc (list[int], optional):
        The values for k the top-k cmc evaluation during testing / evaluation.
        Default [1, 5, 10, 50].
    write_embeds (list[bool, bool], optional):
        Whether to write the embeddings for the Query and Gallery Dataset to the tensorboard writer.
        Only really feasible for smaller datasets ~1k embeddings.
        Default [False, False].
    """

    val_dl: TorchDataLoader
    """The torch DataLoader containing the validation (query) data."""

    # The heart of the project might get a little larger...
    # pylint: disable=too-many-arguments,too-many-locals

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        test_loader: TorchDataLoader,
        val_loader: TorchDataLoader,
        train_loader: TorchDataLoader = None,
    ):
        super().__init__(
            config=config,
            model=model,
            test_loader=test_loader,
            train_loader=train_loader,
        )
        self.val_dl = val_loader

        self.validate_params(test_validations, "params_test")
        self.topk_cmc: list[int] = self.params_test.get("topk_cmc", [1, 5, 10, 50])

        if self.config["is_training"]:
            self.validate_params(train_validations, attrib_name="params_train")

            self.nof_classes: int = self.params_train["nof_classes"]

            self.topk_acc: list[int] = self.params_train.get("topk_acc", [1])

    def get_target(self, ds: DataSample) -> torch.Tensor:
        """Get the target pIDs from the data."""
        return ds["class_id"].long()

    def get_data(self, ds: DataSample) -> torch.Tensor:
        """Get the image crop from the data."""
        return ds["image_crop"]

    @enable_keyboard_interrupt
    @torch.enable_grad()
    def _get_train_loss(self, data: DataSample, _curr_iter: int) -> torch.Tensor:
        _, pred_id_probs = self.model(self.get_data(data))
        target_ids = self.get_target(data)

        loss = self.loss(pred_id_probs, target_ids)

        topk_accuracies = metric.compute_accuracy(prediction=pred_id_probs, target=target_ids, topk=self.topk_acc)
        for k, accu in topk_accuracies.items():
            self.writer.add_scalar(f"Train/top-{k} acc", accu, global_step=_curr_iter)

        return loss

    @enable_keyboard_interrupt
    @torch.no_grad()
    def _extract_data(
        self, dl: TorchDataLoader, model, desc: str, write_embeds: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a dataloader, extract the embeddings describing the people and the target pIDs using the model.
        Additionally, compute the accuracy and send the embeddings to the writer.

        Args:
            dl: The DataLoader to extract the data from.
            model: The (compiled) model to use for predicting the outputs.
            desc: A description for printing, writing, and saving the data.
            write_embeds: Whether to write the embeddings to the tensorboard writer.
                Only "smaller" Datasets should be added.
                Default False.

        Returns:
            embeddings, target_ids
        """

        embed_l: list[torch.Tensor] = []
        t_ids_l: list[torch.Tensor] = []
        imgs_l: list[torch.Tensor] = []

        batch_t: DifferenceTimer = DifferenceTimer()
        batch: DataSample

        for batch_idx, batch in tqdm(enumerate(dl), desc=f"Extract {desc}", total=len(dl)):

            # batch start
            time_batch_start = time.time()  # reset timer for retrieving the data
            curr_iter = (self.curr_epoch - 1) * len(dl) + batch_idx

            # Extract the (cropped) input image and the target pID.
            # Then use the model to compute the predicted embedding and the predicted pID probabilities.
            t_id = self.get_target(batch)
            img_crop = self.get_data(batch)
            embed, _ = model(img_crop)

            # keep the results in lists
            embed_l.append(embed)
            t_ids_l.append(t_id)
            if write_embeds:
                imgs_l.append(img_crop)

            # timing
            batch_t.add(time_batch_start)
            self.writer.add_scalar(f"Test/batch_time_{desc}", batch_t[-1], global_step=curr_iter)
            self.writer.add_scalar(f"Test/indiv_time_{desc}", batch_t[-1] / len(batch), global_step=curr_iter)

        del t_id, embed, img_crop

        # concatenate the result lists
        p_embed: torch.Tensor = torch.cat(embed_l)  # 2D gt embeddings  [N, E]
        t_ids: torch.Tensor = torch.cat(t_ids_l)  # 1D gt person labels [N]
        N: int = len(t_ids)

        assert len(t_ids) == len(p_embed), f"tids: {len(t_ids)}, embed: {len(p_embed)}"

        self.logger.debug(f"{desc} - Shapes - embeddings: {p_embed.shape}, target pIDs: {t_ids.shape}")
        del embed_l, t_ids_l

        # normalize the predicted embeddings if wanted
        p_embed = self._normalize_test(p_embed)

        if write_embeds:
            # write embedding results - take only the first 32x32 due to limitations in tensorboard
            self.logger.info("Add embeddings to writer.")
            self.writer.add_embedding(
                mat=p_embed[: min(512, N), :],
                metadata=t_ids[: min(512, N)].tolist(),
                label_img=torch.cat(imgs_l)[: min(512, N)] if imgs_l else None,  # 4D images [N x C X h x w]
                tag=f"Test/{desc}_embeds_{self.curr_epoch}",
            )

        assert isinstance(p_embed, torch.Tensor), f"p_embed is {p_embed}"
        assert isinstance(t_ids, torch.Tensor), f"t_ids is {t_ids}"

        return p_embed, t_ids

    @torch.no_grad()
    @enable_keyboard_interrupt
    def test(self) -> dict[str, any]:
        r"""Test the embeddings predicted by the model on the Test-DataLoader.

        Compute Rank-N for every rank in params_test["ranks"].
        Compute mean average precision of predicted target labels.
        """
        results: dict[str, any] = {}

        # set up model
        if self.params_test.get("compile_model", False):
            self.logger.debug("Test - Compile the model")
            model = torch.compile(self.model)
        else:
            model = self.model

        if not hasattr(model, "eval"):
            warnings.warn("`model.eval()` is not available.")
        model.eval()  # set model to test / evaluation mode

        start_time: float = time.time()

        self.logger.info(f"#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####")
        self.logger.info("Loading, extracting, and predicting data, this might take a while...")

        q_embed, q_t_ids = self._extract_data(
            dl=self.test_dl,
            model=model,
            desc="Query",
            write_embeds=self.params_test["write_embeds"][0] if "write_embeds" in self.params_test else False,
        )
        g_embed, g_t_ids = self._extract_data(
            dl=self.val_dl,
            model=model,
            desc="Gallery",
            write_embeds=self.params_test["write_embeds"][1] if "write_embeds" in self.params_test else False,
        )

        self.logger.debug("Use metric to compute the distance matrix.")
        distance_matrix = self.metric(q_embed, g_embed)
        self.logger.debug(f"Shape of distance matrix: {distance_matrix.shape}")

        self.logger.debug("Computing CMC")
        results["cmc"] = metric.compute_cmc(
            distmat=distance_matrix,
            query_pids=q_t_ids,
            gallery_pids=g_t_ids,
            ranks=self.topk_cmc,
        )

        self.print_results(results)
        self.write_results(results, prepend="Test")

        self.logger.info(f"Test time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.logger.info(f"#### Evaluation of {self.name} complete ####")

        return results

    def visualize_ranked_results(self, distmat: torch.Tensor) -> None:
        """Use torchreids version of visualizing ranked results"""
        raise NotImplementedError
