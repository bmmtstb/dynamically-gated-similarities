"""
Engine for a full model of the dynamically gated similarity tracker.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine import EngineModule
from dgs.utils.states import DataSample
from dgs.utils.types import Config


class DGSEngine(EngineModule):
    """An engine class for training and testing the dynamically gated similarity tracker.

    For this model:

    - ``get_data()`` should return the DataSample
    - ``get_target()`` should return the target class IDs
    - ``train_dl`` contains the training data as usual
    - ``test_dl`` TODO

    Train Params
    ------------

    Test Params
    -----------

    Optional Test Params
    --------------------

    """

    # The heart of the project might get a little larger...
    # pylint: disable=too-many-arguments,too-many-locals

    model: DGSModule

    def __init__(
        self, config: Config, model: nn.Module, test_loader: TorchDataLoader, train_loader: TorchDataLoader = None
    ):
        super().__init__(config=config, model=model, test_loader=test_loader, train_loader=train_loader)

    def get_data(self, ds: DataSample) -> any:
        return ds

    def get_target(self, ds: DataSample) -> any:
        return ds["class_id"].long()

    def test(self) -> dict[str, any]:
        raise NotImplementedError

    def _get_train_loss(self, data: DataSample, _curr_iter: int) -> torch.Tensor:
        raise NotImplementedError
