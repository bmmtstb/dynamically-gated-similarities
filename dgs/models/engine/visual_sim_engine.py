"""
Engine for visual embedding training and testing.
"""

import time
import warnings
from datetime import timedelta
from typing import Type

import torch
from torch import nn, optim
from torch.nn.functional import softmax as torch_f_softmax
from torch.utils.data import DataLoader as TorchDataLoader
from torcheval.metrics.functional import multiclass_auprc
from tqdm import tqdm

from dgs.models.engine.engine import EngineModule
from dgs.models.metric import compute_cmc
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.states import DataSample, get_ds_data_getter
from dgs.utils.types import Config


class VisualSimilarityEngine(EngineModule):
    """An engine class for training and testing visual similarities using visual embeddings.

    For this model:

    - ``get_data()`` should return the image crop
    - ``get_target()`` should return the target images and the target ids

    """

    val_dl: TorchDataLoader
    """The torch DataLoader containing the validation (query) data."""

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

    def get_target(self, ds: DataSample) -> tuple[torch.Tensor, torch.LongTensor]:
        """Get target IDs"""
        imgs, ids = get_ds_data_getter(["image_crop", "person_id"])(ds)
        return imgs, ids.long()

    def get_data(self, ds: DataSample) -> torch.Tensor:
        """Get the image crop from the data."""
        return get_ds_data_getter(["image_crop"])(ds)[0]

    def _get_train_loss(self, data: DataSample) -> torch.Tensor:
        _, pred_ids = self.model(self.get_data(data))
        _, target_ids = self.get_target(data)
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
        start_time: float = time.time()

        if not hasattr(self.model, "eval"):
            warnings.warn("Neither model.eval() nor model.set_model_mode() are present.")
        self.model.eval()  # set model to test / evaluation mode

        self.print("normal", f"#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####")
        self.print("normal", "Loading and extracting data, this might take a while...")

        t_embeds: list[torch.Tensor] = []
        t_ids: list[torch.LongTensor] = []
        t_pred_ids: list[torch.Tensor] = []
        for batch_gallery in tqdm(self.val_dl, desc="Extract gallery data", leave=False):
            # extract target data, as image and target_id, then use the model to compute the target embedding
            t_imgs, t_id = self.get_target(batch_gallery)
            t_embed, t_pred_id = self.model(t_imgs)
            t_embeds.append(t_embed)
            t_ids.append(t_id)
            t_pred_ids.append(torch_f_softmax(t_pred_id, dim=1))

        targ_embed: torch.Tensor = torch.cat(t_embeds)  # 2D gt embeddings   [num_classes, E]
        targ_ids: torch.LongTensor = torch.cat(t_ids).long()  # 1D gt person labels [n_gallery]
        self.print("debug", f"Shapes - target embedding: {targ_embed.shape}, target IDs: {targ_ids.shape}")

        self.print("debug", "Computing mAP - Gallery")
        results: dict[str, any] = {
            "mean_avg_precision_gallery": multiclass_auprc(
                input=torch.cat(t_pred_ids),  # class predictions - probabilities with shape [n_gallery x num_classes]
                target=targ_ids,  # 1D LongTensor of ground truth labels with shape [n_gallery].
            ).item(),
        }
        self.print("debug", f"mAP - Gallery: {results['mean_avg_precision_gallery']}")
        del t_embed, t_embeds, t_id, t_ids, t_pred_id, t_pred_ids

        # extract the data for query and gallery dataloader
        p_embeds: list[torch.Tensor] = []
        p_ids: list[torch.Tensor] = []
        p_targ_ids: list[torch.LongTensor] = []
        for batch_query in tqdm(self.test_dl, desc="Extract query data", leave=False):
            # extract data and use the current model to get a prediction
            p_imgs, p_targ_id = self.get_target(batch_query)
            p_embed, p_id = self.model(p_imgs)
            p_embeds.append(p_embed)
            p_ids.append(p_id)
            p_targ_ids.append(p_targ_id)

        pred_embed: torch.Tensor = torch.cat(p_embeds)  # 2D predict embeddings  [n_query x E]
        pred_id_probs: torch.Tensor = torch_f_softmax(torch.cat(p_ids), dim=1)  # [n_query x num_classes]
        self.print("debug", f"Shapes - predicted id probabilities: {pred_id_probs.shape}")
        # sample 1 id from ``[n_query x num_classes]`` 1D predicted person IDs [n_query]
        pred_ids: torch.LongTensor = torch.multinomial(pred_id_probs, num_samples=1).squeeze_().long()
        self.print("debug", f"Shapes - predicted embedding: {pred_embed.shape}, predicted IDs: {pred_ids.shape}")

        self.print("debug", "Computing mAP - Query")
        results["mean_avg_precision_query"] = multiclass_auprc(
            input=pred_id_probs,  # class predictions - probabilities with shape [n_query x num_classes]
            target=torch.cat(p_targ_ids).long(),  # 1D LongTensor of ground truth labels with shape [n_query].
        ).item()
        self.print("debug", f"mAP - Query: {results['mean_avg_precision_query']}")
        del p_embed, p_embeds, p_id, p_ids

        pred_embed, targ_embed = self._normalize_test(pred_embed, targ_embed)

        self.print("normal", "Testing predicted Embeddings and IDs")
        self.print("debug", "Computing distance matrix")

        distance_matrix = self.metric(pred_embed, targ_embed)

        self.print("debug", f"Shape of distance matrix: {distance_matrix.shape}")
        self.print("debug", "Computing CMC")

        results["cmc"] = compute_cmc(
            distmat=distance_matrix,
            labels=targ_ids,
            predictions=pred_ids,
            ranks=self.params_test.get("ranks", [1, 5, 10, 20]),
        )

        self.print_results(results)
        self.write_results(results, prepend="Test", index=self.curr_epoch)

        self.print("normal", f"Test time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.print("normal", f"#### Evaluation of {self.name} complete ####")

        return results
