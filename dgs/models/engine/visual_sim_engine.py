"""
Engine for training and testing visual embedding modules.

Notes:
    Kind of obsolete, due to being able to use the engines from |torchreid|_ to train visual embedding models.
"""

import time
from datetime import timedelta

import torch as t
from torch.utils.data import DataLoader as TDataLoader
from tqdm import tqdm

from dgs.models.engine.engine import EngineModule
from dgs.models.metric import get_metric, metric, METRICS
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.similarity.torchreid import TorchreidVisualSimilarity
from dgs.utils.config import DEF_VAL
from dgs.utils.state import State
from dgs.utils.timer import DifferenceTimer
from dgs.utils.types import Config, Metric, Results, Validations

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
    "image_key": ["optional", str],
}


class VisualSimilarityEngine(EngineModule):
    """An engine class for training and testing visual similarities using visual embeddings.

    For this model:

    - ``get_data()`` should return the image crop
    - ``get_target()`` should return the target class IDs
    - ``train_dl`` contains the training data as usual
    - ``test_dl`` contains the query data
    - ``val_dl`` contains the gallery data


    Train Params
    ------------

    nof_classes (int):
        The number of classes in the training set.


    Test Params
    -----------

    metric (str|callable):
        The name or class of the metric used during testing / evaluation.
        The metric in the ``VisualSimilarityEngine`` is only used
        to compute the distance between the query and gallery embeddings.
        Therefore, a distance-based metric should be used.

        It is possible to pass additional initialization kwargs to the metric
        by adding them to the ``metric_kwargs`` parameter.

    Optional Train Params
    ---------------------

    topk_acc (list[int], optional):
        The values for k for the top-k accuracy evaluation during training.
        Default ``DEF_VAL.engine.visual.topk_acc``.

    Optional Test Params
    --------------------

    metric_kwargs (dict, optional):
        Specific kwargs for the metric.
        Default ``DEF_VAL.engine.visual.metric_kwargs``.
    topk_cmc (list[int], optional):
        The values for k the top-k cmc evaluation during testing / evaluation.
        Default ``DEF_VAL.engine.visual.topk_cmc``.
    write_embeds (list[bool, bool], optional):
        Whether to write the embeddings for the Query and Gallery Dataset to the tensorboard writer.
        Only really feasible for smaller datasets ~1k embeddings.
        Default ``DEF_VAL.engine.visual.write_embeds``.
    image_key (str, optional):
        Which key to use when loading the image from the state in :meth:`get_data`.
        Default ``DEF_VAL.engine.visual.image_key``.
    """

    # The heart of the project might get a little larger...
    # pylint: disable=too-many-arguments

    val_dl: TDataLoader
    """The torch DataLoader containing the validation (query) data."""

    model: TorchreidVisualSimilarity

    metric: Metric
    """A metric function used to compute the embedding distance."""

    def __init__(
        self,
        config: Config,
        model: TorchreidVisualSimilarity,
        test_loader: TDataLoader,
        val_loader: TDataLoader,
        *,
        train_loader: TDataLoader = None,
        **kwargs,
    ):
        super().__init__(config=config, model=model, test_loader=test_loader, train_loader=train_loader, **kwargs)
        self.val_dl = val_loader

        self.validate_params(test_validations, "params_test")
        self.topk_cmc: list[int] = self.params_test.get("topk_cmc", DEF_VAL["engine"]["visual"]["topk_cmc"])

        # get metric and kwargs
        self.metric = get_metric(self.params_test["metric"])(
            **self.params_test.get("metric_kwargs", DEF_VAL["engine"]["visual"]["metric_kwargs"])
        )

        self.image_key: str = self.params_test.get("image_key", DEF_VAL["engine"]["visual"]["image_key"])

        if self.is_training:
            self.validate_params(train_validations, attrib_name="params_train")

            self.nof_classes: int = self.params_train["nof_classes"]

            self.topk_acc: list[int] = self.params_train.get("topk_acc", DEF_VAL["engine"]["visual"]["topk_acc"])

    def get_target(self, ds: State) -> t.Tensor:
        """Get the target pIDs from the data."""
        return ds["class_id"].long()

    def get_data(self, ds: State) -> t.Tensor:
        """Get the image crop or other requested image from the state."""
        return ds[self.image_key]

    @enable_keyboard_interrupt
    def _get_train_loss(self, data: State, _curr_iter: int) -> t.Tensor:

        target_ids = self.get_target(data)

        crops = self.get_data(data)
        pred_id_probs = self.model.predict_ids(crops)

        loss = self.loss(pred_id_probs, target_ids)

        topk_accuracies = metric.compute_accuracy(prediction=pred_id_probs, target=target_ids, topk=self.topk_acc)
        self.writer.add_scalars(
            main_tag="Train/acc",
            tag_scalar_dict={str(k): v for k, v in topk_accuracies.items()},
            global_step=_curr_iter,
        )

        return loss

    @t.no_grad()
    @enable_keyboard_interrupt
    def _extract_data(self, dl: TDataLoader, desc: str, write_embeds: bool = False) -> tuple[t.Tensor, t.Tensor]:
        """Given a dataloader, extract the embeddings describing the people and the target pIDs using the model.
        Additionally, compute the accuracy and send the embeddings to the writer.

        Args:
            dl: The DataLoader to extract the data from.
            desc: A description for printing, writing, and saving the data.
            write_embeds: Whether to write the embeddings to the tensorboard writer.
                Only "smaller" Datasets should be added.
                Default False.

        Returns:
            embeddings, target_ids
        """

        embed_l: list[t.Tensor] = []
        t_ids_l: list[t.Tensor] = []
        imgs_l: list[t.Tensor] = []

        batch_t: DifferenceTimer = DifferenceTimer()
        batch: State

        for batch_idx, batch in tqdm(enumerate(dl), desc=f"Extract {desc}", total=len(dl)):

            # batch start
            time_batch_start = time.time()  # reset timer for retrieving the data
            curr_iter = (self.curr_epoch - 1) * len(dl) + batch_idx

            # Extract the (cropped) input image and the target pID.
            # Then use the model to compute the predicted embedding and the predicted pID probabilities.
            t_id = self.get_target(batch)
            img_crop = self.get_data(batch)
            embed = self.model.get_data(batch)

            # keep the results in lists
            embed_l.append(embed)
            t_ids_l.append(t_id)
            if write_embeds:
                imgs_l.append(img_crop)

            # timing
            batch_t.add(time_batch_start)
            self.writer.add_scalars(
                main_tag="Test/time",
                tag_scalar_dict={f"batch_{desc}": batch_t[-1], f"indiv_{desc}": batch_t[-1] / len(batch)},
                global_step=curr_iter,
            )

        del t_id, embed, img_crop

        # concatenate the result lists
        p_embed: t.Tensor = t.cat(embed_l)  # 2D gt embeddings  [N, E]
        t_ids: t.Tensor = t.cat(t_ids_l)  # 1D gt person labels [N]
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
                label_img=t.cat(imgs_l)[: min(512, N)] if imgs_l else None,  # 4D images [N x C x h x w]
                tag=f"Test/{desc}_embeds_{self.curr_epoch}",
            )

        assert isinstance(p_embed, t.Tensor), f"p_embed is {p_embed}"
        assert isinstance(t_ids, t.Tensor), f"t_ids is {t_ids}"

        return p_embed, t_ids

    @t.no_grad()
    def test(self) -> dict[str, any]:
        r"""Test the embeddings predicted by the model on the Test-DataLoader.

        Compute Rank-N for every rank in ``self.topk_cmc``.
        Compute mean average precision of predicted target labels.
        """
        results: dict[str, any] = {}

        self.set_model_mode("eval")

        start_time: float = time.time()

        self.logger.info(f"#### Start Evaluating {self.name} - Epoch {self.curr_epoch} ####")
        self.logger.info("Loading, extracting, and predicting data, this might take a while...")

        q_embed, q_t_ids = self._extract_data(
            dl=self.test_dl,
            desc="Query",
            write_embeds=self.params_test.get("write_embeds", DEF_VAL["engine"]["visual"]["write_embeds"])[0],
        )
        g_embed, g_t_ids = self._extract_data(
            dl=self.val_dl,
            desc="Gallery",
            write_embeds=self.params_test.get("write_embeds", DEF_VAL["engine"]["visual"]["write_embeds"])[1],
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
        # DUPLICATE #
        results["cmc_inv"] = metric.compute_cmc(
            distmat=self.metric(g_embed, q_embed),
            query_pids=g_t_ids,
            gallery_pids=q_t_ids,
            ranks=self.topk_cmc,
        )

        self.print_results(results)
        self.write_results(results, prepend="Test")

        self.logger.info(f"Test time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.logger.info(f"#### Evaluation of {self.name} complete ####")

        return results

    def evaluate(self) -> Results:
        raise NotImplementedError

    @t.no_grad()
    def predict(self) -> t.Tensor:
        """Predict the visual embeddings for the test data.

        Notes:
            Depending on the number of predictions (``N``) and the embeddings size (``E``),
            the resulting tensor(s) can get incredibly huge.
            The prediction for the validation data of the |PT21| dataset is roughly 300MB.

        Returns:
            torch.Tensor: The predicted embeddings as tensor of shape: ``[N x E]``
        """
        self.set_model_mode("eval")
        start_time: float = time.time()
        self.logger.info(f"#### Start Prediction {self.name} ####")

        embeds, _ = self._extract_data(
            dl=self.test_dl,
            desc="Predict",
            write_embeds=self.params_test.get("write_embeds", DEF_VAL["engine"]["visual"]["write_embeds"])[0],
        )
        self.logger.info(f"Predict time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.logger.info(f"#### Prediction of {self.name} complete ####")

        return embeds
