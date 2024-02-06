"""
Engine for visual embedding training and testing.
"""

import time
import warnings
from datetime import timedelta
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader as TorchDataLoader
from torcheval.metrics.functional import multiclass_auprc
from tqdm import tqdm

from dgs.models.engine.engine import EngineModule
from dgs.models.metric import compute_cmc
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.states import DataSample, get_ds_data_getter
from dgs.utils.types import Config, Validations

train_validations: Validations = {
    "nof_classes": ["int", ("gt", 0)],
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
            config, model, test_loader, train_loader=train_loader, test_only=test_only, lr_scheds=lr_scheds
        )
        self.val_dl = val_loader

        if not self.test_only:
            self.validate_params(train_validations, attrib_name="params_train")

            self.nof_classes: int = self.params_train["nof_classes"]

    def get_target(self, ds: DataSample) -> torch.LongTensor:
        """Get the target pIDs from the data."""
        return get_ds_data_getter(["person_id"])(ds)[0].long()

    def get_data(self, ds: DataSample) -> torch.Tensor:
        """Get the image crop from the data."""
        return get_ds_data_getter(["image_crop"])(ds)[0]

    def _get_train_loss(self, data: DataSample) -> torch.Tensor:
        _, pred_ids = self.model(self.get_data(data))
        target_ids = self.get_target(data)

        assert all(
            tid <= self.nof_classes for tid in target_ids
        ), f"{set(tid.item() for tid in target_ids if tid > self.nof_classes)}"

        oh_t_ids = F.one_hot(target_ids, self.nof_classes).float()  # pylint: disable=not-callable

        assert pred_ids.shape == oh_t_ids.shape, f"p: {pred_ids.shape} t: {oh_t_ids.shape}"
        assert pred_ids.dtype == oh_t_ids.dtype, f"p: {pred_ids.dtype} t: {oh_t_ids.dtype}"

        loss = self.loss(pred_ids, target_ids)
        return loss

    @torch.no_grad()
    @enable_keyboard_interrupt
    def test(self) -> dict[str, any]:
        r"""Test the embeddings predicted by the model on the Test-DataLoader.

        Compute Rank-N for every rank in params_test["ranks"].
        Compute mean average precision of predicted target labels.

        Cumulative Matching Characteristics
        -----------------------------------

        For further information see: https://cysu.github.io/open-reid/notes/evaluation_metrics.html.

        The `single-gallery-shot` CMC top-k accuracy is defined as

        .. math::
           Acc_k = \begin{cases}
              1 & \text{if top-}k\text{ ranked gallery samples contain the query identity} \\
              0 & \text{otherwise}
           \end{cases}

        This represents a shifted step function.
        The final CMC curve is computed by averaging the shifted step functions over all the queries.

        The `multi-gallery-shot` accuracy is not implemented.
        """
        results: dict[str, any] = {}

        def obtain_test_data(dl: TorchDataLoader, desc: str) -> tuple[torch.Tensor, torch.LongTensor]:
            """Given a dataloader,
            extract the embeddings describing the people, target pIDs, and the pIDs the model predicted.

            Args:
                dl: The DataLoader to extract the data from.
                desc: A description for printing and saving the data.

            Returns:
                embeddings, target_ids
            """

            embed_l: list[torch.Tensor] = []
            t_ids_l: list[torch.LongTensor] = []
            m_ap_l: list[torch.Tensor] = []

            for batch in tqdm(dl, desc=f"Extract {desc} data"):  # with N = len(dl)
                # Extract the (cropped) input image and the target pID.
                # Then use the model to compute the predicted embedding and the predicted pID probabilities.
                t_imgs = self.get_data(batch)
                t_id = self.get_target(batch)
                embed, pred_id_prob = self.model(t_imgs)

                # Obtain class probability predictions and mAP from data
                m_ap = multiclass_auprc(
                    input=F.softmax(pred_id_prob, dim=1),  # 2D class probabilities [B, num_classes]
                    target=t_id,  # gt labels    [B]
                    average=None,  # due to batches, we need to compute the mean later...
                )

                # keep the results in lists
                embed_l.append(embed)
                t_ids_l.append(t_id)
                m_ap_l.append(m_ap)

            del t_imgs, t_id, embed, pred_id_prob, m_ap

            # concatenate the result lists
            p_embed: torch.Tensor = torch.cat(embed_l)  # 2D gt embeddings             [N, E]
            t_ids: torch.LongTensor = torch.cat(t_ids_l).long()  # 1D gt person labels [N]
            m_aps: torch.Tensor = torch.cat(m_ap_l)  # 1D mAP for every class          [N]

            assert (
                len(t_ids) == len(p_embed) == len(m_aps)
            ), f"t ids: {len(t_ids)}, p embed: {len(p_embed)}, mAPs: {len(m_aps)}"

            self.print(
                "debug",
                f"{desc} - Shapes - embeddings: {p_embed.shape}, target pIDs: {t_ids.shape}",
            )
            del embed_l, t_ids_l, m_ap_l

            # normalize the predicted embeddings if wanted
            p_embed = self._normalize(p_embed)

            # concat all the intermediate mAPs and compute the unweighted mean
            total_m_ap: float = m_aps.mean().item()
            results[f"mean_avg_precision_{desc.lower()}"] = total_m_ap
            self.print("debug", f"mAP - {desc}: {total_m_ap:.2}")

            return p_embed, t_ids

        start_time: float = time.time()

        if not hasattr(self.model, "eval"):
            warnings.warn("Neither model.eval() nor model.set_model_mode() are present.")
        self.model.eval()  # set model to test / evaluation mode

        self.print("normal", f"\n#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####\n")
        self.print("normal", "Loading, extracting, and predicting data, this might take a while...")

        g_embed, g_t_ids = obtain_test_data(dl=self.val_dl, desc="Gallery")
        q_embed, q_t_ids = obtain_test_data(dl=self.test_dl, desc="Query")

        self.print("debug", "Computing distance matrix")
        distance_matrix = self.metric(q_embed, g_embed)
        self.print("debug", f"Shape of distance matrix: {distance_matrix.shape}")

        self.print("debug", "Computing CMC")
        results["cmc"] = compute_cmc(
            distmat=distance_matrix,
            query_pids=q_t_ids,
            gallery_pids=g_t_ids,
            ranks=self.params_test.get("ranks", [1, 5, 10, 20]),
        )

        self.print_results(results)
        self.write_results(results, prepend="Test", index=self.curr_epoch)

        self.print("normal", f"Test time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.print("normal", f"\n#### Evaluation of {self.name} complete ####\n")

        return results
