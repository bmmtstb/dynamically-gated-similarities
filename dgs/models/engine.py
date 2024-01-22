"""
Class and functions used during training and testing of different modules.
"""
import math
import os
import time
from datetime import date
from typing import Callable, Union

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
from dgs.utils.torchtools import save_checkpoint
from dgs.utils.types import Config, FilePath, Validations

train_validations: Validations = {
    "epochs": ["int", ("gte", 1)],
    "loss": [("or", (("callable", ...), ("in", LOSS_FUNCTIONS)))],
    "metric": [("or", (("callable", ...), ("in", METRICS)))],
    "optimizer": [("or", (("callable", ...), ("in", OPTIMIZERS)))],
    "log_dir": [("or", (("folder exists in project", ...), ("folder exists", ...)))],
    "ranks": ["optional", "iterable"],
}

test_validations: Validations = {
    "metric": [("or", (("callable", ...), ("in", METRICS)))],
    "test_normalize": ["optional", "bool"],
    "ranks": ["optional", "iterable"],
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

    ...

    Optional Train Params
    ---------------------

    epochs (int, optional):
        The number of epochs to run the training for.
        Default 1.
    log_dir (FilePath, optional):
        Path to directory where all the files of this run are saved.
        Default "./results/"
    ranks (list[int], optional):
        Which ranks to compute during the evaluation.
        Default [1, 5, 10, 20]
    test_normalize (bool, optional):
        Whether to normalize the prediction and targets before the evaluation.
        Default False.
    """

    # The engine is the heart of most algorithms and therefore contains a los of stuff.
    # pylint: disable = too-many-instance-attributes, too-many-arguments

    loss: nn.Module
    metric: nn.Module
    optimizer: optim.Optimizer
    model: nn.Module
    writer: SummaryWriter
    test_dl: TorchDataLoader
    train_dl: TorchDataLoader

    lr_sched: list[optim.lr_scheduler.LRScheduler]
    """The learning-rate sheduler(s) can be changed by setting ``engine.lr_scheduler = [..., ...]``."""

    get_data: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]]
    """Function to retrieve the data used in the model's prediction from the train- and test- DataLoaders."""

    get_target: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]]
    """Function to retrieve the evaluation targets from the train- and test- DataLoaders."""

    def __init__(
        self,
        config: Config,
        test_loader: TorchDataLoader,
        get_data: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]],
        get_target: Callable[[DataSample], Union[torch.Tensor, tuple[torch.Tensor, ...]]],
        train_loader: TorchDataLoader = None,
        test_only: bool = False,
    ):
        super().__init__(config, [])
        self.validate_params(test_validations)
        self.params_test: Config = get_sub_config(config, ["test"])
        self.params_train: Config = {}
        if not test_only:
            self.validate_params(train_validations)
            self.params_train = get_sub_config(config, ["train"])

        self.test_dl = test_loader
        self.train_dl = train_loader
        self.get_data = get_data
        self.get_target = get_target

        self.epochs: int = self.params_train.get("epochs", 1)
        self.curr_epoch: int = 0
        self.log_dir: FilePath = self.params_train.get("log_dir", "./results/")

        self.loss = get_loss_function(self.params["loss"])(**self.params.get("loss_kwargs", {}))
        self.metric = get_metric(self.params["metric"])(**self.params.get("metric_kwargs", {}))
        # the optimizer needs some model params to be set up
        self.optimizer = get_optimizer(self.params["optimizer"])(**self.params.get("optim_kwargs", {}))
        # the learning-rate scheduler needs the optimizer
        self.lr_sched = [optim.lr_scheduler.ConstantLR(optimizer=self.optimizer)]
        self.writer = SummaryWriter(log_dir=self.log_dir, **self.params.get("writer_kwargs", {}))

    @enable_keyboard_interrupt
    def __call__(self, *args, **kwargs) -> any:
        return self.run(*args, **kwargs)

    @enable_keyboard_interrupt
    def run(self) -> None:
        """Run the model. First train, then test!"""
        if self.print("normal"):
            print(f"#### Starting run {self.name} ####")
        if self.print("normal") and "description" in self.config:
            print(f"Config Description: {self.config['description']}")
        self.train()
        self.test()

    @enable_keyboard_interrupt
    def test(self) -> any:
        """Test model on target dataset(s). Compute Rank-1."""
        self.model.eval()  # set model to test / evaluation mode
        if self.print("normal"):
            print(f"#### Start Evaluating {self.name} ####")
            print("Loading and extracting data, this might take a while...")

        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch_data in tqdm(self.test_dl, desc="Extract test data", leave=False, position=1):
            # extract data and use the current model to get a prediction
            preds.append(self.model(self.get_data(batch_data)).to(self.device))
            # extract target data
            targets.append(self.get_target(batch_data).to(self.device))

        pred: torch.Tensor = torch.cat(preds)
        target: torch.Tensor = torch.cat(targets)
        del preds, targets

        if self.print("debug"):
            print(f"prediction shape: {pred.shape}, target: {target.shape}")

        if self.params["test_normalize"]:
            if self.print("debug"):
                print("Normalizing test data")
            pred: torch.Tensor = nn.functional.normalize(pred)
            target: torch.Tensor = nn.functional.normalize(target)

        if self.print("debug"):
            print("Computing distance matrix")
        distance_matrix = self.metric(pred, target)

        if self.print("debug"):
            print("Computing CMC and mAP")

        cmc, m_ap = ([], distance_matrix)  # fixme evaluate rank!
        if not len(cmc):
            raise NotImplementedError

        print(f"#### Results - Epoch {self.curr_epoch} ####")
        print(f"mAP: {m_ap:.1%}")
        print("CMC curve:")
        for r in self.params.get("ranks", [1, 5, 10, 20]):
            print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")

        # at the end use the writer to save results
        self.writer.add_scalar(f"Test/{self.name}/rank1", cmc[0], self.curr_epoch)
        self.writer.add_scalar(f"Test/{self.name}/mAP", m_ap, self.curr_epoch)

        if self.print("normal"):
            print("#### Evaluation complete ####")

        return cmc[0]

    @enable_keyboard_interrupt
    def train(self) -> None:
        """Train the given model using the given loss function, optimizer, and learning-rate scheduler.

        After every epoch, the current model is tested and the current model is saved.
        """
        if self.train_dl is None:
            raise ValueError("No DataLoader for the Training data was given. Can't continue.")
        self.model.train()  # set model to train mode
        if self.print("normal"):
            print("#### Start Training ####")

        # initialize variables
        losses: list[float] = []
        epoch_times: DifferenceTimer = DifferenceTimer()
        batch_times: DifferenceTimer = DifferenceTimer()
        data_times: DifferenceTimer = DifferenceTimer()
        num_batches: int = math.ceil(len(self.train_dl) / self.train_dl.batch_size)
        data: DataSample

        for self.curr_epoch in tqdm(range(self.epochs), desc="Epoch", position=0):
            epoch_loss = 0
            loss = None  # init for tqdm text
            time_epoch_start = time.time()
            time_batch_start = time.time()  # reset timer for retrieving the data

            # loop over all the data
            for batch_idx, data in tqdm(
                enumerate(self.train_dl),
                desc=f"Per Batch - "
                f"last loss: {loss.item() if loss else ''} - "
                f"lr: {self.optimizer.param_groups[-1]['lr']}",
                position=1,
                leave=False,
            ):
                data_times.add(time_batch_start)

                # OPTIMIZE MODEL
                self.optimizer.zero_grad()
                output = self.model(self.get_data(data))
                loss = self.loss(output, self.get_target(data))
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
                # ############ #
                # END OF BATCH #
                # ############ #
                time_batch_start = time.time()  # reset timer for retrieving the data before entering next loop

            # ############ #
            # END OF EPOCH #
            # ############ #
            epoch_times.add(time_epoch_start)
            # handle updating the learning rate scheduler(s)
            for sched in self.lr_sched:
                sched.step()
            # save loss and learned model data / weights
            losses.append(epoch_loss)
            self.save_model(self.curr_epoch, self.test())  # does also call self.test() !!
            if self.print("debug"):
                print(f"Training: epoch {self.curr_epoch} loss: {epoch_loss}")
                print(f"Training: epoch {self.curr_epoch} time: {round(epoch_times[-1])} [s]")

        # ############### #
        # END OF TRAINING #
        # ############### #

        if self.print("normal"):
            print(data_times.print(name="data", prepend="Training"))
            print(batch_times.print(name="batch", prepend="Training"))
            print(epoch_times.print(name="epoch", prepend="Training", hms=True))
            print("#### Training complete ####")

        self.writer.close()

    def save_model(self, epoch: int, rank1) -> None:
        """Save the current model and other weights into a '.pth' file.

        Args:
            epoch: The epoch this model is saved.
            rank1: Rank-1 accuracy is a performance metric used in deep learning to evaluate the model's accuracy.
                It measures whether the top prediction matches the ground truth label for a given sample.
        """

        for sched in self.lr_sched:
            save_checkpoint(
                state={
                    "state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "rank1": rank1,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": sched.state_dict(),
                },
                save_dir=os.path.join(
                    self.log_dir, f"./checkpoints/{self.name}_{str(sched.get_lr())}_{date.today().strftime('%Y%m%d')}/"
                ),
                verbose=self.print("normal"),
            )

    def terminate(self) -> None:
        """Handle forceful termination, e.g., ctrl+c"""
        self.writer.close()
