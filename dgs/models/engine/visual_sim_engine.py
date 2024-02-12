"""
Engine for visual embedding training and testing.
"""

import time
import warnings
from datetime import timedelta
from typing import Type

import torch
from torch import nn, optim
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from dgs.models.engine.engine import EngineModule
from dgs.models.metric import compute_accuracy, compute_cmc
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.states import DataSample, get_ds_data_getter
from dgs.utils.types import Config, Validations

train_validations: Validations = {
    "nof_classes": [int, ("gt", 0)],
    # optional
    "topk": ["optional", tuple, ("forall", [int, ("gt", 0)])],
}

test_validations: Validations = {
    # optional
    "topk": ["optional", tuple, ("forall", [int, ("gt", 0)])]
}


class VisualSimilarityEngine(EngineModule):
    """An engine class for training and testing visual similarities using visual embeddings.

    For this model:

    - ``get_data()`` should return the image crop
    - ``get_target()`` should return the target pIDs

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
        test_only: bool = False,
        lr_scheds: list[Type[optim.lr_scheduler.LRScheduler]] = None,
    ):
        super().__init__(
            config=config,
            model=model,
            test_loader=test_loader,
            train_loader=train_loader,
            test_only=test_only,
            lr_scheds=lr_scheds,
        )
        self.val_dl = val_loader

        self.validate_params(test_validations, "params_test")
        self.test_topk: tuple[int, ...] = self.params_train.get("topk", (1, 5, 10, 50))

        if not self.test_only:
            self.validate_params(train_validations, attrib_name="params_train")

            self.nof_classes: int = self.params_train["nof_classes"]

            self.train_topk: tuple[int, ...] = self.params_train.get("topk", (1, 5, 10, 50))

    def get_target(self, ds: DataSample) -> torch.Tensor:
        """Get the target pIDs from the data."""
        return get_ds_data_getter(["person_id"])(ds)[0].long()

    def get_data(self, ds: DataSample) -> torch.Tensor:
        """Get the image crop from the data."""
        return get_ds_data_getter(["image_crop"])(ds)[0]

    def _get_train_loss(self, data: DataSample, _curr_iter: int) -> torch.Tensor:
        _, pred_id_probs = self.model(self.get_data(data))
        target_ids = self.get_target(data)

        assert all(
            tid <= self.nof_classes for tid in target_ids
        ), f"{set(tid.item() for tid in target_ids if tid > self.nof_classes)}"

        oh_t_ids = self._ids_to_one_hot(ids=target_ids, nof_classes=self.nof_classes)

        assert pred_id_probs.shape == oh_t_ids.shape, f"p: {pred_id_probs.shape} t: {oh_t_ids.shape}"
        # assert pred_id_probs.dtype == oh_t_ids.dtype, f"p: {pred_id_probs.dtype} t: {oh_t_ids.dtype}"

        # loss = self.loss(pred_id_probs, oh_t_ids)
        loss = self.loss(pred_id_probs, target_ids)

        topk_accuracies = compute_accuracy(prediction=pred_id_probs, target=target_ids, topk=self.train_topk)
        for k, accu in topk_accuracies.items():
            self.writer.add_scalar(f"Train/top-{k} acc", accu, global_step=_curr_iter)

        return loss

    @torch.no_grad()
    @enable_keyboard_interrupt
    def test(self) -> dict[str, any]:
        r"""Test the embeddings predicted by the model on the Test-DataLoader.

        Compute Rank-N for every rank in params_test["ranks"].
        Compute mean average precision of predicted target labels.
        """
        results: dict[str, any] = {}

        def obtain_test_data(dl: TorchDataLoader, desc: str) -> tuple[torch.Tensor, torch.Tensor]:
            """Given a dataloader,
            extract the embeddings describing the people, target pIDs, and the pIDs the model predicted.

            Args:
                dl: The DataLoader to extract the data from.
                desc: A description for printing and saving the data.

            Returns:
                embeddings, target_ids
            """

            total_m_aps: dict[int, float] = {k: 0 for k in self.test_topk}
            embed_l: list[torch.Tensor] = []
            t_ids_l: list[torch.Tensor] = []

            for batch in tqdm(dl, desc=f"Extract {desc} data"):  # with N = len(dl)
                # Extract the (cropped) input image and the target pID.
                # Then use the model to compute the predicted embedding and the predicted pID probabilities.
                t_imgs = self.get_data(batch)
                t_id = self.get_target(batch)
                embed, pred_id_prob = self.model(t_imgs)

                # Obtain class probability predictions and mAP from data
                B = t_imgs.size(0)
                m_aps: dict[int, float] = compute_accuracy(
                    prediction=pred_id_prob,  # 2D class probabilities [B, num_classes]
                    target=t_id,  # gt labels    [B]
                    topk=self.test_topk,
                )
                for k in self.test_topk:
                    total_m_aps[k] += m_aps[k] * float(B)  # map*B, later we will div by total N

                # keep the results in lists
                embed_l.append(embed)
                t_ids_l.append(t_id)

            del t_imgs, t_id, embed, pred_id_prob, m_aps

            # concatenate the result lists
            p_embed: torch.Tensor = torch.cat(embed_l)  # 2D gt embeddings             [N, E]
            t_ids: torch.Tensor = torch.cat(t_ids_l)  # 1D gt person labels [N]

            for k, val in total_m_aps.items():
                m_ap = val / len(dl.dataset)
                results[f"top-{k} acc"] = m_ap
                self.logger.debug(f"top-{k} acc: {m_ap:.2}")

            assert len(t_ids) == len(p_embed), f"t ids: {len(t_ids)}, p embed: {len(p_embed)}"

            self.logger.debug(f"{desc} - Shapes - embeddings: {p_embed.shape}, target pIDs: {t_ids.shape}")
            del embed_l, t_ids_l, total_m_aps

            # normalize the predicted embeddings if wanted
            p_embed = self._normalize(p_embed)

            # concat all the intermediate mAPs and compute the unweighted mean

            return p_embed, t_ids

        start_time: float = time.time()

        if not hasattr(self.model, "eval"):
            warnings.warn("`model.eval()` is not available.")
        self.model.eval()  # set model to test / evaluation mode

        self.logger.info(f"\n#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####\n")
        self.logger.info("Loading, extracting, and predicting data, this might take a while...")

        g_embed, g_t_ids = obtain_test_data(dl=self.val_dl, desc="Gallery")
        q_embed, q_t_ids = obtain_test_data(dl=self.test_dl, desc="Query")

        results["query_embed"] = q_embed

        self.logger.debug("Use metric to compute the distance matrix.")
        distance_matrix = self.metric(q_embed, g_embed)
        self.logger.debug(f"Shape of distance matrix: {distance_matrix.shape}")

        self.logger.debug("Computing CMC")
        results["cmc"] = compute_cmc(
            distmat=distance_matrix,
            query_pids=q_t_ids,
            gallery_pids=g_t_ids,
            ranks=self.params_test.get("ranks", [1, 5, 10, 20]),
        )

        self.print_results(results)
        self.write_results(results, prepend="Test", index=self.curr_epoch)

        self.logger.info(f"Test time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.logger.info(f"\n#### Evaluation of {self.name} complete ####\n")

        return results

    def visualize_ranked_results(self, distmat: torch.Tensor) -> None:
        """Use torchreids version of visualizing ranked results"""
        raise NotImplementedError
