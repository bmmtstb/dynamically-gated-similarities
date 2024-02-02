"""
Class and functions used during training and testing of different modules.
"""

import math
import os
import time
import warnings
from abc import abstractmethod
from datetime import date
from typing import Type

import torch
from torch import nn, optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dgs.models.loss import get_loss_function, LOSS_FUNCTIONS
from dgs.models.metric import get_metric, METRICS
from dgs.models.module import BaseModule, enable_keyboard_interrupt
from dgs.models.optimizer import get_optimizer, OPTIMIZERS
from dgs.models.states import DataSample
from dgs.utils.config import get_sub_config
from dgs.utils.timer import DifferenceTimer
from dgs.utils.torchtools import resume_from_checkpoint, save_checkpoint
from dgs.utils.types import Config, FilePath, Validations

train_validations: Validations = {
    "loss": [("or", (("callable", ...), ("in", LOSS_FUNCTIONS)))],
    "optimizer": [("or", (("callable", ...), ("in", OPTIMIZERS)))],
    # optional
    "epochs": ["optional", "int", ("gte", 1)],
    "optimizer_kwargs": ["optional", "dict"],
    "loss_kwargs": ["optional", "dict"],
}

test_validations: Validations = {
    "metric": [("or", (("callable", ...), ("in", METRICS)))],
    # optional
    "log_dir": ["optional", ("or", (("folder exists in project", ...), ("folder exists", ...)))],
    "test_normalize": ["optional", "bool"],
    "ranks": ["optional", "iterable", ("all type", int)],
    "writer_kwargs": ["optional", "dict"],
}


class EngineModule(BaseModule):
    """Module for training, validating, and testing other Modules.

    Most of the settings are defined within the configuration file in the `training` section.

    Notes:
        The trained module is saved every epoch.

    Test Params
    -----------

    metric ():
        ...


    Train Params
    ------------

    loss ():
        ...
    optimizer ():
        ...

    Optional Test Params
    --------------------

    log_dir (FilePath, optional):
        Path to directory where all the files of this run are saved.
        Default "./results/"
    metric_kwargs (dict, optional):
        Additional kwargs for the metric.
        Default {}.
    ranks (list[int], optional):
        The cmc ranks to use for evaluation.
        This value is used during training and testing.
        Default [1, 5, 10, 20]
    test_normalize (bool, optional):
        Whether to normalize the prediction and target during testing.
        Default False.
    writer_kwargs (dict, optional):
        Additional kwargs for the torch writer.
        Default {}.


    Optional Train Params
    ---------------------

    epochs (int, optional):
        The number of epochs to run the training for.
        Default 1.
    optimizer_kwargs (dict, optional):
        Additional kwargs for the optimizer.
        Default {}.
    loss_kwargs (dict, optional):
        Additional kwargs for the loss.
        Default {}.
    """

    # The engine is the heart of most algorithms and therefore contains a los of stuff.
    # pylint: disable = too-many-instance-attributes, too-many-arguments

    loss: nn.Module
    metric: nn.Module
    optimizer: optim.Optimizer
    model: nn.Module
    writer: SummaryWriter

    test_dl: TorchDataLoader
    """The torch DataLoader containing the test data."""

    train_dl: TorchDataLoader
    """The torch DataLoader containing the training data."""

    lr_sched: list[optim.lr_scheduler.LRScheduler]
    """The learning-rate sheduler(s) can be changed by setting ``engine.lr_scheduler = [..., ...]``."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        test_loader: TorchDataLoader,
        train_loader: TorchDataLoader = None,
        test_only: bool = False,
        lr_scheds: list[Type[optim.lr_scheduler.LRScheduler]] = None,
    ):
        super().__init__(config, [])

        # Set up general attributes
        self.curr_epoch: int = 0
        self.test_only = test_only
        self.model = model

        # Set up test attributes
        self.params_test: Config = get_sub_config(config, ["test"])
        self.validate_params(test_validations, attrib_name="params_test")
        self.test_dl = test_loader
        self.metric = get_metric(self.params_test["metric"])(**self.params_test.get("metric_kwargs", {}))

        self.log_dir: FilePath = self.params_test.get("log_dir", "./results/")
        self.writer = SummaryWriter(log_dir=self.log_dir, **self.params_test.get("writer_kwargs", {}))

        # Set up train attributes
        self.params_train: Config = {}
        if not self.test_only:
            self.params_train = get_sub_config(config, ["train"])
            self.validate_params(train_validations, attrib_name="params_train")
            if train_loader is None:
                raise ValueError("test_only is False but train_loader is None.")
            self.train_dl = train_loader
            self.epochs: int = self.params_train.get("epochs", 1)
            self.start_epoch: int = self.params_train.get("start_epoch", 0)
            self.loss = get_loss_function(self.params_train["loss"])(
                **self.params_train.get("loss_kwargs", {})  # optional loss kwargs
            )
            self.optimizer = get_optimizer(self.params_train["optimizer"])(
                self.model.parameters(),
                **self.params_train.get("optimizer_kwargs", {"lr": 0.001}),  # optional optimizer kwargs
            )
            # the learning-rate scheduler needs the optimizer, therefore, it will be initialized here.
            if lr_scheds is None:
                lr_scheds = [optim.lr_scheduler.ConstantLR]
            self.lr_sched = [lr_sched(optimizer=self.optimizer) for lr_sched in lr_scheds]

    @enable_keyboard_interrupt
    def __call__(self, *args, **kwargs) -> any:
        return self.run(*args, **kwargs)

    @abstractmethod
    def get_data(self, ds: DataSample) -> any:
        """Function to retrieve the data used in the model's prediction from the train- and test- DataLoaders."""
        raise NotImplementedError

    @abstractmethod
    def get_target(self, ds: DataSample) -> any:
        """Function to retrieve the evaluation targets from the train- and test- DataLoaders."""
        raise NotImplementedError

    @enable_keyboard_interrupt
    def run(self) -> None:
        """Run the model. First train, then test!"""
        if self.can_print("normal"):
            print(f"#### Starting run {self.name} ####")
        if self.can_print("normal") and "description" in self.config:
            print(f"Config Description: {self.config['description']}")

        if not self.test_only:
            self.train()

        self.test()

    @abstractmethod
    @enable_keyboard_interrupt
    def test(self) -> dict[str, any]:
        """Run tests, defined in Sub-Engine.

        Returns:
            dict[str, any]: A dictionary containing all the computed metrics.
        """
        raise NotImplementedError

    @enable_keyboard_interrupt
    def train(self) -> None:
        """Train the given model using the given loss function, optimizer, and learning-rate scheduler.

        After every epoch, the current model is tested and the current model is saved.
        """
        if self.train_dl is None:
            raise ValueError("No DataLoader for the Training data was given. Can't continue.")

        if self.can_print("normal"):
            print("#### Start Training ####")

        # set model to train mode
        self.model.train()

        # initialize variables
        losses: list[float] = []
        epoch_times: DifferenceTimer = DifferenceTimer()
        batch_times: DifferenceTimer = DifferenceTimer()
        data_times: DifferenceTimer = DifferenceTimer()
        num_batches: int = math.ceil(len(self.train_dl) / self.train_dl.batch_size)
        data: DataSample

        for self.curr_epoch in tqdm(range(self.start_epoch, self.epochs), desc="Epoch", position=1):
            epoch_loss = 0
            loss = None  # init for tqdm text
            time_epoch_start = time.time()
            time_batch_start = time.time()  # reset timer for retrieving the data

            # loop over all the data
            for batch_idx, data in tqdm(
                enumerate(self.train_dl),
                desc=f"Per Batch - "
                f"last loss: {str(loss.item()) if loss and hasattr(loss, 'item') else ''} - "
                f"lr: {self.optimizer.param_groups[-1]['lr']:.8}",
                position=0,
                leave=False,
            ):
                data_times.add(time_batch_start)

                # OPTIMIZE MODEL
                self.optimizer.zero_grad()
                loss = self._get_train_loss(data)
                loss.backward()
                self.optimizer.step()
                # OPTIMIZE END

                batch_times.add(time_batch_start)
                epoch_loss += loss.item()
                curr_iter = self.curr_epoch * num_batches + batch_idx
                self.writer.add_scalar("Train/loss", loss.item(), curr_iter)
                self.writer.add_scalar("Train/batch_time", batch_times[-1], curr_iter)
                self.writer.add_scalar("Train/data_time", data_times[-1], curr_iter)
                self.writer.add_scalar("Train/lr", self.optimizer.param_groups[-1]["lr"], curr_iter)
                self.writer.flush()
                # ############ #
                # END OF BATCH #
                # ############ #
                time_batch_start = time.time()  # reset timer for retrieving the data before entering next loop

            # ############ #
            # END OF EPOCH #
            # ############ #
            epoch_times.add(time_epoch_start)
            losses.append(epoch_loss)
            # handle updating the learning rate scheduler(s)
            for sched in self.lr_sched:
                sched.step()
            # evaluate current model
            metrics = self.test()
            self.save_model(epoch=self.curr_epoch, metrics=metrics)
            if self.can_print("debug"):
                print(f"Training: epoch {self.curr_epoch} loss: {epoch_loss}")
                print(f"Training: epoch {self.curr_epoch} time: {round(epoch_times[-1])} [s]")

        # ############### #
        # END OF TRAINING #
        # ############### #

        if self.can_print("normal"):
            print(data_times.print(name="data", prepend="Training"))
            print(batch_times.print(name="batch", prepend="Training"))
            print(epoch_times.print(name="epoch", prepend="Training", hms=True))
            print("#### Training complete ####")

        self.writer.close()

    def save_model(self, epoch: int, metrics: dict[str, any]) -> None:  # pragma: no cover
        """Save the current model and other weights into a '.pth' file.

        Args:
            epoch: The epoch this model is saved.
            metrics: A dict containing the computed metrics for this module.
        """
        curr_lr = self.optimizer.param_groups[-1]["lr"]

        save_checkpoint(
            state={
                "model": self.model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": {i: sched.state_dict() for i, sched in enumerate(self.lr_sched)},
            },
            save_dir=os.path.join(
                self.log_dir, f"./checkpoints/{self.name}_{str(curr_lr)}_{date.today().strftime('%Y%m%d')}/"
            ),
            verbose=self.can_print("normal"),
        )

    def load_model(self, path: FilePath) -> None:  # pragma: no cover
        """Load the model from a file. Set the start epoch to the epoch specified in the loaded model."""
        self.start_epoch = resume_from_checkpoint(
            fpath=path, model=self.model, optimizer=self.optimizer, scheduler=self.lr_sched
        )

    def terminate(self) -> None:  # pragma: no cover
        """Handle forceful termination, e.g., ctrl+c"""
        if hasattr(self, "writer"):
            self.writer.flush()
            self.writer.close()
        for attr in ["model", "optimizer", "lr_sched", "test_dl", "train_dl", "val_dl", "metric", "loss"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def _normalize_test(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """If ``params_test.test_normalize`` is True, return the normalized prediction and target."""
        if self.params_test.get("test_normalize", False):
            if self.can_print("debug"):
                print("Normalizing test data")
            pred: torch.Tensor = nn.functional.normalize(pred)
            target: torch.Tensor = nn.functional.normalize(target)
        return pred, target

    @abstractmethod
    def _get_train_loss(self, data: DataSample) -> torch.Tensor:  # pragma: no cover
        """Compute the loss during training given the data.

        Different models can have different outputs and a different number of targets.
        This function should get overwritten by subclasses to ensure correct behavior.
        A default implementation that only has one output and one target can be found here.
        """
        raise NotImplementedError

    def write_results(self, results: dict[str, any], prepend: str, index: int) -> None:
        """Given a dictionary of results, use the writer to save the values."""
        for key, value in results.items():
            if isinstance(value, (int, float, str)):
                self.writer.add_scalar(f"{prepend}/{self.name}/{key}", value, index)
            elif isinstance(value, torch.Tensor) and value.ndim == 1 and value.size(0) == 1:
                self.writer.add_scalar(f"{prepend}/{self.name}/{key}", value.item(), index)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.writer.add_scalar(f"{prepend}/{self.name}/{key}-{sub_key}", sub_value, index)
            elif isinstance(value, (list, dict, set)):
                for i, sub_value in enumerate(value):
                    self.writer.add_scalar(f"{prepend}/{self.name}/{key}-{i}", sub_value, index)
            else:
                warnings.warn(f"Unknown result for writer: {value} {key}")
        if self.can_print("debug"):
            print("results have been written to writer")
        self.writer.flush()

    def print_results(self, results: dict[str, any]) -> None:
        """Given a dictionary of results, print them to the console if allowed."""
        if self.can_print("normal"):
            print(f"#### Results - Epoch {self.curr_epoch} ####")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.2%}")
                elif key.lower() == "cmc":
                    print("CMC curve:")
                    for r, cmc_i in value.items():
                        print(f"Rank-{r}: {cmc_i:.1%}")
                else:
                    warnings.warn(f"Unknown result for printing: {key} {value}")
