"""
Engine for visual embedding training and testing.
"""

import torch
from torcheval.metrics.functional import multiclass_auprc
from tqdm import tqdm

from dgs.models.engine.engine import EngineModule
from dgs.models.metric import compute_cmc
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.states import DataSample, get_ds_data_getter


class VisualEmbeddingEngine(EngineModule):
    """An engine class for training and testing visual embedding modules.

    For this model:

    - `get_data()` should return the image crop
    - `get_target()` should return the target images and the target ids

    """

    def get_target(self, ds: DataSample) -> tuple[torch.Tensor, torch.IntTensor]:
        """Get target IDs"""
        imgs, ids = get_ds_data_getter(["image_crop", "person_id"])(ds)
        return imgs, ids.int()

    def get_data(self, ds: DataSample) -> torch.Tensor:
        """Get the image crop from the data."""
        return get_ds_data_getter(["image_crop"])(ds)[0]

    @enable_keyboard_interrupt
    def _get_train_loss(self, data: DataSample) -> torch.Tensor:
        _, pred_ids = self.model(*self.get_data(data))
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
        self.model.eval()  # set model to test / evaluation mode
        if self.print("normal"):
            print(f"#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####")
            print("Loading and extracting data, this might take a while...")

        # extract the data

        p_embeds: list[torch.Tensor] = []
        p_ids: list[torch.Tensor] = []
        t_embeds: list[torch.Tensor] = []
        t_ids: list[torch.IntTensor] = []

        for batch_data in tqdm(self.test_dl, desc="Extract test data", leave=False, position=1):
            # extract data and use the current model to get a prediction
            p_embed, p_id = self.model(self.get_data(batch_data))
            p_embeds.append(p_embed)
            p_ids.append(p_id)

            # extract target data, as image and target_id, then use the model to compute the target embedding
            t_imgs, t_id = self.get_target(batch_data)
            t_embed, _ = self.model(t_imgs)
            t_embeds.append(t_embed)
            t_ids.append(t_id)

        pred_embed: torch.Tensor = torch.cat(p_embeds)  # 2D predict embeddings  [n_samples, E]
        targ_embed: torch.Tensor = torch.cat(t_embeds)  # 2D gt embeddings   [n_class, E]
        # sample 1 id from ``[n_samples x num_classes]`` 1D predicted person IDs [n_samples]
        pred_ids: torch.IntTensor = torch.multinomial(torch.cat(p_ids), num_samples=1).squeeze_().int()
        targ_ids: torch.IntTensor = torch.cat(t_ids).int()  # 1D gt person labels [n_samples]
        del p_embed, p_embeds, p_id, p_ids, t_embed, t_embeds, t_id, t_ids

        if self.print("debug"):
            print(f"Shape of predicted embedding: {pred_embed.shape}, shape of embedding target: {targ_embed.shape}")

        pred_embed, targ_embed = self._normalize_test(pred_embed, targ_embed)

        if self.print("normal"):
            print("Testing predicted Embeddings and IDs")

        if self.print("debug"):
            print("Computing distance matrix")
        distance_matrix = self.metric(pred_embed, targ_embed)

        if self.print("debug"):
            print("Computing CMC and mAP")

        results: dict[str, any] = {
            "mean_avg_precision": multiclass_auprc(
                input=distance_matrix,
                # tensor label predictions - probabilities or logits with shape [n_sample x n_class]
                target=targ_ids,  # 1D IntTensor of ground truth labels with shape [n_samples (x 1)].
            ),
            "cmc": compute_cmc(
                distmat=distance_matrix,
                labels=targ_ids,
                predictions=pred_ids,
                ranks=self.params_test.get("ranks", [1, 5, 10, 20]),
            ),
        }

        self.print_results(results)
        self.write_results(results, prepend="Test", index=self.curr_epoch)

        if self.print("normal"):
            print(f"#### Evaluation of {self.name} complete ####")

        return results
