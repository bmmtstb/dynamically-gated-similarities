"""
Engine for training and testing similarity models and the respective alpha weights.
"""

import time
from datetime import timedelta

import torch as t
from torch.utils.data import DataLoader as TDataLoader

from dgs.models.engine import EngineModule
from dgs.models.metric import get_metric, METRICS
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.similarity.similarity import SimilarityModule
from dgs.utils.config import DEF_VAL
from dgs.utils.state import State
from dgs.utils.types import Config, Metric, Results, Validations

train_validations: Validations = {
    # optional
}

test_validations: Validations = {
    "metric": [("or", [[str, ("in", METRICS.keys())], "callable"])],
    # optional
    "metric_kwargs": ["optional", dict],
}


class SimilarityEngine(EngineModule):
    """An engine for training and testing similarity models independently.

    For this model:

    - ``get_data()`` should return the same as this similarity functions get_data call
    - ``get_target()`` should return the  same as this similarity functions get_target call
    - ``train_dl`` contains the training data as usual
    - ``test_dl`` contains the test data
    - ``val_dl`` contains the validation data

    Train Params
    ------------

    Test Params
    -----------

    metric (str|callable):
        The name or class of the metric used during testing / evaluation.

        It is possible to pass additional initialization kwargs to the metric
        by adding them to the ``metric_kwargs`` parameter.


    Optional Train Params
    ---------------------

    acc_k_train (list[int], optional):
        A list of values used during training to check whether the accuracy lies within a margin of k percent.
        Default ``DEF_VAL.engine.sim.acc_k_train``.

    Optional Test Params
    --------------------

    acc_k_test (list[int], optional):
        A list of values used during test to check whether the accuracy lies within a margin of k percent.
        Default ``DEF_VAL.engine.sim.acc_k_test``.
    metric_kwargs (dict, optional):
        Specific kwargs for the metric.
        Default ``DEF_VAL.engine.sim.metric_kwargs``.
    """

    # The heart of the project might get a little larger...
    # pylint: disable=too-many-arguments

    model: SimilarityModule

    metric: Metric
    """A metric function used to compute the embedding distance."""

    val_dl: TDataLoader
    """The torch DataLoader containing the validation data."""

    def __init__(
        self,
        config: Config,
        model: SimilarityModule,
        test_loader: TDataLoader,
        val_loader: TDataLoader,
        train_loader: TDataLoader = None,
        **kwargs,
    ):
        super().__init__(config=config, model=model, test_loader=test_loader, train_loader=train_loader, **kwargs)

        self.val_dl = val_loader

        # Params - Test
        self.validate_params(test_validations, "params_test")
        self.metric = get_metric(self.params_test["metric"])(
            **self.params_test.get("metric_kwargs", DEF_VAL["engine"]["visual"]["metric_kwargs"])
        )
        # Params - Train

    def get_data(self, ds: State) -> t.Tensor:
        """Use the similarity model to get the data."""
        return self.model.get_data(ds)

    def get_target(self, ds: State) -> any:
        """Use the similarity model to get the target data."""
        return self.model.get_target(ds)

    def test(self) -> Results:
        r"""Test whether the predicted alpha probability (:math:`\alpha_{\mathrm{pred}}`)
        matches the number of correct predictions (:math:`\alpha_{\mathrm{correct}}`)
        divided by the total number of predictions (:math:`N`).

        With :math:`\alpha{\mathrm{pred}} = \frac{\alpha_{\mathrm{correct}}}{N}`
        :math`\alpha{\mathrm{pred}}` is counted as correct if
        :math:`\alpha{\mathrm{pred}}-k \leq \alpha{\mathrm{correct}} \leq \alpha{\mathrm{pred}}+k`.

        Returns:
            Results dict containing the Accuracy ("acc-k") as the number of correct predictions within k percent.
        """
        self.logger.debug("Start Test - set model to eval mode")
        results: dict[str, any] = {}

        self.set_model_mode("eval")

        start_time: float = time.time()
        self.print_results(results)
        self.write_results(results, prepend="Test")

        self.logger.info(f"Test time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.logger.info(f"#### Evaluation of {self.name} complete ####")
        return results

    def predict(self) -> any:
        """Predict the weighted similarity between the data and the target."""
        self.logger.debug("Start Predict - set model to eval mode")
        self.set_model_mode("eval")
        start_time: float = time.time()

        # ...

        self.logger.info(f"Predict time total: {str(timedelta(seconds=round(time.time() - start_time)))}")
        self.logger.info(f"#### Prediction of {self.name} complete ####")

    @enable_keyboard_interrupt
    def _get_train_loss(self, data: State, _curr_iter: int) -> t.Tensor:
        target_ids = self.get_target(data)

        crops = self.get_data(data)
        pred_id_probs = self.model.predict_ids(crops)

        loss = self.loss(pred_id_probs, target_ids)
        return loss
