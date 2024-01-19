"""
Class and functions used during training and testing of different modules.
"""
import math
import os
import time
from datetime import date, timedelta
from typing import Callable, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dgs.models.loss import get_loss_function, LOSS_FUNCTIONS
from dgs.models.module import BaseModule, enable_keyboard_interrupt
from dgs.models.optimizer import get_optimizer, OPTIMIZERS
from dgs.models.states import DataSample
from dgs.utils.torchtools import save_checkpoint
from dgs.utils.types import Config, FilePath, NodePath, Validations

module_validations: Validations = {
    "batch_size": ["int", ("gte", 1)],
    "epochs": ["int", ("gte", 1)],
    "loss": [("or", (("callable", ...), ("in", LOSS_FUNCTIONS)))],
    "metric": [],
    "optimizer": [("or", (("callable", ...), ("in", OPTIMIZERS)))],
    "log_dir": [("or", (("folder exists in project", ...), ("folder exists", ...)))],
}


class EngineModule(BaseModule):
    """Module for training, validating, and testing other Modules.

    Most of the settings are defined within the configuration file in the `training` section.

    Methods:
        train: Train the given `nn.Module`
        test: Test the given `nn.Module`
        run: First train the given `nn.Module`, then test it.

    Notes:
        The trained module is saved every epoch.
    """

    loss: nn.Module
    metric: nn.Module
    optimizer: optim.Optimizer
    model: nn.Module
    writer: SummaryWriter
    test_dl: TorchDataLoader
    train_dl: TorchDataLoader

    get_data: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]]
    get_target: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]]

    def __init__(
        self,
        config: Config,
        path: NodePath,
        test_loader: TorchDataLoader,
        get_data: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]],
        get_target: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]],
        train_loader: TorchDataLoader = None,
    ):
        super().__init__(config, path)
        self.validate_params(module_validations)

        self.test_dl = test_loader
        self.train_dl = train_loader
        self.get_data = get_data
        self.get_target = get_target

        self.epochs: int = self.params["epochs"]
        self.curr_epoch: int = 0
        self.log_dir: FilePath = self.params["log_dir"]
        self.batch_size: int = self.params["batch_size"]

        self.loss = get_loss_function(self.params["loss"])(**self.params.get("loss_kwargs", {}))
        self.metric = ...
        # optimizer needs model params
        self.optimizer = get_optimizer(self.params["optimizer"])(**self.params.get("optim_kwargs", {}))
        self.writer = SummaryWriter(log_dir=self.log_dir, **self.params.get("writer_kwargs", {}))

    @enable_keyboard_interrupt
    def __call__(self, *args, **kwargs) -> any:
        return self.run(*args, **kwargs)

    @enable_keyboard_interrupt
    def run(self) -> None:
        """Run the model. First train, then test!"""

    @enable_keyboard_interrupt
    def test(self) -> any:
        """Test model on target dataset(s). Compute Rank-1."""
        self.model.eval()  # set model to test / evaluation mode

        if self.print("normal"):
            print(f"##### Evaluating {self.name}  #####")
            print("Loading and extracting data")

        pred: list[torch.Tensor] = []
        target: list[torch.Tensor] = []
        for batch_data in tqdm(self.test_dl, desc="Extract data", leave=False):
            pred.append(self.model(self.get_data(batch_data)).to(self.device))
            target.append(self.get_target(batch_data).to(self.device))

        pred: torch.Tensor = torch.cat(pred)
        target: torch.Tensor = torch.cat(target)

        rank1, mAP = (1, 1)  # todo create evaluation
        # at the end use the writer to save results
        self.writer.add_scalar(f"Test/{self.name}/rank1", rank1, self.curr_epoch)
        self.writer.add_scalar(f"Test/{self.name}/mAP", mAP, self.curr_epoch)

        return rank1

    @enable_keyboard_interrupt
    def train(self) -> None:
        """Train the given model."""
        self.model.train()  # set model to train mode

        # initialize variables
        losses: list[float] = []
        batch_time: list[float] = []
        data_time: list[float] = []
        time_start = time.time()
        num_batches: int = math.ceil(len(self.train_dl) / self.batch_size)
        data: DataSample

        end = time.time()
        for self.curr_epoch in tqdm(range(self.epochs), desc="Epoch", position=0):
            epoch_loss = 0

            # loop over all the data
            for batch_idx, data in tqdm(enumerate(self.train_dl), desc="Per Batch", position=1, leave=False):
                data_time.append(time.time() - end)
                self.optimizer.zero_grad()
                output = self.model(self.get_data(data))
                loss = self.loss(output, self.get_target(data))
                loss.backward()
                self.optimizer.step()

                batch_time.append(time.time() - end)
                epoch_loss += loss.item()
                curr_iter = self.curr_epoch * num_batches + batch_idx
                self.writer.add_scalar("Train/loss", loss.item(), curr_iter)
                self.writer.add_scalar("Train/batch_time", batch_time[-1], curr_iter)
                self.writer.add_scalar("Train/data_time", data_time[-1], curr_iter)

            # handle the end of one epoch
            losses.append(epoch_loss)
            self.save_model(self.curr_epoch, self.test())
            end = time.time()  # reset timer for data

        if self.print("normal"):
            elapsed = str(timedelta(seconds=round(time.time() - time_start)))
            print(f"Elapsed {elapsed}")

        self.writer.close()

    def save_model(self, epoch: int, rank1) -> None:
        """

        Args:
            epoch: The epoch this model is saved.
            rank1: Rank-1 accuracy is a performance metric used in deep learning to evaluate the model's accuracy.
                It measures whether the top prediction matches the ground truth label for a given sample.
        """

        save_checkpoint(
            {
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "rank1": rank1,
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.log_dir, f"./checkpoints/{self.name}/{date.today().strftime('%Y%m%d')}/"),
        )

    def terminate(self) -> None:
        """Handle forceful termination, e.g., ctrl+c"""
        self.writer.close()
