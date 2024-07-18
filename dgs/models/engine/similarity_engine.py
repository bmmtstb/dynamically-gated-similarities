"""
Engine for training and testing similarity models and the respective alpha weights.
"""

import time
from datetime import timedelta

import torch as t
from torch.utils.data import DataLoader as TDataLoader

from dgs.models.engine import EngineModule
from dgs.models.metric import METRICS
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

    - ``get_data()`` should return the image crop
    - ``get_target()`` should return ???
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

    Optional Test Params
    --------------------

    metric_kwargs (dict, optional):
        Specific kwargs for the metric.
        Default ``DEF_VAL.engine.sim.metric_kwargs``.

    """

    # The heart of the project might get a little larger...
    # pylint: disable=too-many-arguments

    model: SimilarityModule

    metric: Metric
    """A metric function used to compute the embedding distance."""

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

        # Params - Test
        self.validate_params(test_validations, "params_test")
        self.image_key: str = self.params_test.get("image_key", DEF_VAL["engine"]["visual"]["image_key"])

    def get_data(self, ds: State) -> t.Tensor:
        """Use the similarity model to get the data."""
        return self.model.get_data(ds)

    def get_target(self, ds: State) -> any:
        """Use the similarity model to get the target data."""
        return self.model.get_target(ds)

    def test(self) -> Results:
        """Test whether the predicted alpha probability matches the number of correct predictions
        divided by the total number of predictions.

        Returns:
            Results dict containing the Accuracy ("acc-k") as the number of correct predictions within k percent.
        """
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
