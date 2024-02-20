"""
Class and functions used during training and testing of different modules.
"""

import logging
import os
import time
import warnings
from abc import abstractmethod
from typing import Union

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch._dynamo import OptimizedModule
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import tv_tensors
from tqdm import tqdm

from dgs.models.loss import get_loss_function, LOSS_FUNCTIONS
from dgs.models.metric import get_metric
from dgs.models.module import BaseModule, enable_keyboard_interrupt
from dgs.models.optimizer import get_optimizer, OPTIMIZERS
from dgs.models.scheduler import get_scheduler, SCHEDULERS
from dgs.utils.config import get_sub_config
from dgs.utils.exceptions import InvalidConfigException
from dgs.utils.states import DataSample
from dgs.utils.timer import DifferenceTimer
from dgs.utils.torchtools import resume_from_checkpoint, save_checkpoint
from dgs.utils.types import Config, FilePath, Validations
from dgs.utils.visualization import torch_show_image

train_validations: Validations = {
    "loss": [("any", ["callable", ("in", LOSS_FUNCTIONS.keys())])],
    "optimizer": [("any", ["callable", ("in", OPTIMIZERS.keys())])],
    # optional
    "epochs": ["optional", int, ("gte", 1)],
    "loss_kwargs": ["optional", dict],
    "optimizer_kwargs": ["optional", dict],
    "save_interval": ["optional", int, ("gte", 1)],
    "scheduler": ["optional", ("any", ["callable", ("in", SCHEDULERS.keys())])],
    "scheduler_kwargs": ["optional", dict],
}

test_validations: Validations = {
    # optional
    "compile_model": ["optional", bool],
    "normalize": ["optional", bool],
    "ranks": ["optional", "iterable", ("all type", int)],
    "writer_kwargs": ["optional", dict],
}


class EngineModule(BaseModule):
    """Module for training, validating, and testing other Modules.

    Most of the settings are defined within the configuration file in the `training` section.

    Notes:
        The trained module is saved every epoch.

    Test Params
    -----------



    Train Params
    ------------

    loss (str|callable):
        The name or class of the loss function used to compute the loss during training.
        It is possible to pass additional initialization kwargs to the loss
        by adding them to the ``loss_kwargs`` parameter.

    optimizer (str|callable):
        The name or class of the optimizer used for optimizing the model based on the loss during training.
        It is possible to pass additional initialization kwargs to the optimizer
        by adding them to the ``optimizer_kwargs`` parameter.


    Optional Test Params
    --------------------

    ranks (list[int], optional):
        The cmc ranks to use for evaluation.
        This value is used during training and testing.
        Default [1, 5, 10, 20]
    normalize (bool, optional):
        Whether to normalize the prediction and target during testing.
        Default False.
    writer_kwargs (dict, optional):
        Additional kwargs for the torch writer.
        Default {}.
    compile_model (bool, optional):
        Whether to ``torch.compile`` the given model for testing.
        Requires a SOTA GPU.
        Default False.


    Optional Train Params
    ---------------------

    epochs (int, optional):
        The number of epochs to run the training for.
        Default 1.
    optimizer_kwargs (dict, optional):
        Additional kwargs for the optimizer.
        Default {}.
    scheduler (str|callable, optional):
        The name or instance of a scheduler.
        If you want to use different or multiple schedulers, you can chain them using
        ``torch.optim.lr_scheduler.ChainedScheduler`` or create a custom Scheduler and register it.
        Default "StepLR".
    scheduler_kwargs (dict, optional):
        Additional kwargs for the scheduler.
        Keep in mind that the different schedulers need fairly different kwargs.
        The optimizer will be passed to the scheduler during initialization as the `optimizer` keyword argument.
        Default {"step_size": 1, "gamma": 0.1}.
    loss_kwargs (dict, optional):
        Additional kwargs for the loss.
        Default {}.
    save_interval (int, optional):
        The interval for saving (and evaluating) the model during training.
        Default 5.
    compile_model (bool, optional):
        Whether to ``torch.compile`` the given model for training.
        Requires a SOTA GPU.
        Default False.

    """

    # The engine is the heart of most algorithms and therefore contains a los of stuff.
    # pylint: disable = too-many-instance-attributes

    loss: nn.Module
    metric: nn.Module
    optimizer: optim.Optimizer
    model: Union[nn.Module, OptimizedModule]
    writer: SummaryWriter

    curr_epoch: int = 0

    test_dl: TorchDataLoader
    """The torch DataLoader containing the test data."""

    train_dl: TorchDataLoader
    """The torch DataLoader containing the training data."""

    lr_sched: optim.lr_scheduler.LRScheduler
    """The learning-rate sheduler(s) can be changed by setting ``engine.lr_scheduler = [..., ...]``."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        test_loader: TorchDataLoader,
        train_loader: TorchDataLoader = None,
    ):
        super().__init__(config, [])

        # Set up test attributes
        self.params_test: Config = get_sub_config(config, ["test"])
        self.validate_params(test_validations, attrib_name="params_test")
        self.test_dl = test_loader
        self.metric = get_metric(self.params_test["metric"])(**self.params_test.get("metric_kwargs", {}))

        # Set up general attributes
        self.model = model

        # Logging
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            comment=self.config.get("description"),
            **self.params_test.get("writer_kwargs", {}),
        )
        self.writer.add_scalar("Test/batch_size", self.test_dl.batch_size)

        # Set up train attributes
        self.params_train: Config = {}
        if self.config["is_training"]:
            self.params_train = get_sub_config(config, ["train"])
            self.validate_params(train_validations, attrib_name="params_train")
            if train_loader is None:
                raise InvalidConfigException("is_training is turned on but train_loader is None.")
            # data loader
            self.train_dl = train_loader

            # epochs
            self.epochs: int = self.params_train.get("epochs", 1)
            self.start_epoch: int = self.params_train.get("start_epoch", 1)
            self.curr_epoch = self.start_epoch
            self.save_interval: int = self.params_train.get("save_interval", 5)

            # modules
            self.loss = get_loss_function(self.params_train["loss"])(
                **self.params_train.get("loss_kwargs", {})  # optional loss kwargs
            )
            self.optimizer = get_optimizer(self.params_train["optimizer"])(
                self.model.parameters(),
                **self.params_train.get("optimizer_kwargs", {"lr": 1e-4}),  # optional optimizer kwargs
            )
            # the learning-rate schedulers need the optimizer for instantiation
            self.lr_sched = get_scheduler(self.params_train.get("scheduler", "StepLR"))(
                optimizer=self.optimizer, **self.params_train.get("scheduler_kwargs", {"step_size": 1, "gamma": 0.1})
            )
            self.writer.add_scalar("Train/batch_size", self.test_dl.batch_size)

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
        self.logger.info(f"#### Starting run {self.name} ####")
        if "description" in self.config:
            self.logger.info(f"Config Description: {self.config['description']}")

        if self.config["is_training"]:
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
        """Train the given model using the given loss function, optimizer, and learning-rate schedulers.

        After every epoch, the current model is tested and the current model is saved.
        """
        if self.train_dl is None:
            raise ValueError("No DataLoader for the Training data was given. Can't continue.")

        self.logger.info("#### Start Training ####")

        # set model to train mode
        if not hasattr(self.model, "train"):
            warnings.warn("`model.train()` is not available.")
        self.model.train()

        # compile model if wanted
        if self.params_train.get("compile_model", False):
            self.logger.debug("Train - Compile the model")
            self.model = torch.compile(self.model)

        # initialize variables
        losses: list[float] = []
        epoch_t: DifferenceTimer = DifferenceTimer()
        batch_t: DifferenceTimer = DifferenceTimer()
        data_t: DifferenceTimer = DifferenceTimer()
        data: DataSample

        for self.curr_epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc="Epoch", position=1):
            self.logger.info(f"#### Training - Epoch {self.curr_epoch} ####")

            epoch_loss = 0
            time_epoch_start = time.time()
            time_batch_start = time.time()  # reset timer for retrieving the data

            # loop over all the data
            for batch_idx, data in tqdm(
                enumerate(self.train_dl),
                desc=f"Per Batch - lr: {self.optimizer.param_groups[-1]['lr']:.8}",
                position=2,
                total=len(self.train_dl),
            ):
                curr_iter = (self.curr_epoch - 1) * len(self.train_dl) + batch_idx
                data_t.add(time_batch_start)

                # OPTIMIZE MODEL
                loss = self._get_train_loss(data, curr_iter)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # OPTIMIZE END

                batch_t.add(time_batch_start)
                epoch_loss += loss.item()
                self.writer.add_scalar("Train/loss", loss.item(), global_step=curr_iter)
                self.writer.add_scalar("Train/batch_time", batch_t[-1], global_step=curr_iter)
                self.writer.add_scalar("Train/indiv_time", batch_t[-1] / len(data), global_step=curr_iter)
                self.writer.add_scalar("Train/data_time", data_t[-1], global_step=curr_iter)
                self.writer.add_scalar("Train/lr", self.optimizer.param_groups[-1]["lr"], global_step=curr_iter)
                self.writer.flush()
                # ############ #
                # END OF BATCH #
                # ############ #
                time_batch_start = time.time()  # reset timer for retrieving the data before entering next loop

            # ############ #
            # END OF EPOCH #
            # ############ #
            epoch_t.add(time_epoch_start)
            losses.append(epoch_loss)
            self.logger.info(f"Training: epoch {self.curr_epoch} loss: {epoch_loss:.2}")
            self.logger.info(epoch_t.print(name="epoch", prepend="Training", hms=True))

            # handle updating the learning rate schedulers(s)
            for sched in self.lr_sched:
                sched.step()

            if self.curr_epoch % self.save_interval == 0:
                # evaluate current model every few epochs
                metrics = self.test()
                self.save_model(epoch=self.curr_epoch, metrics=metrics)

        # ############### #
        # END OF TRAINING #
        # ############### #

        self.logger.info(data_t.print(name="data", prepend="Training"))
        self.logger.info(batch_t.print(name="batch", prepend="Training"))
        self.logger.info(epoch_t.print(name="epoch", prepend="Training", hms=True))
        self.logger.info("#### Training complete ####")

        self.writer.flush()

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
            save_dir=os.path.join(self.log_dir, f"./checkpoints/{self.name.replace(' ', '_')}_{curr_lr:.10}/"),
            verbose=self.logger.isEnabledFor(logging.INFO),
        )

    def load_model(self, path: FilePath) -> None:  # pragma: no cover
        """Load the model from a file. Set the start epoch to the epoch specified in the loaded model."""
        self.start_epoch = resume_from_checkpoint(
            fpath=path,
            model=self.model,
            optimizer=self.optimizer if hasattr(self, "optimizer") else None,
            schedulers=self.lr_sched if hasattr(self, "lr_sched") else None,
            verbose=self.logger.isEnabledFor(logging.DEBUG),
        )
        self.curr_epoch = self.start_epoch

    def terminate(self) -> None:  # pragma: no cover
        """Handle forceful termination, e.g., ctrl+c"""
        if hasattr(self, "writer"):
            self.writer.flush()
            self.writer.close()
        for attr in ["model", "optimizer", "lr_sched", "test_dl", "train_dl", "val_dl", "metric", "loss"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def _normalize_test(self, tensor: torch.Tensor) -> torch.Tensor:
        """If ``params_test.normalize`` is True, we want to obtain the normalized prediction and target."""
        if self.params_test.get("normalize", False):
            self.logger.debug("Normalizing test data")
            tensor: torch.Tensor = nn.functional.normalize(tensor)
        return tensor

    @abstractmethod
    def _get_train_loss(self, data: DataSample, _curr_iter: int) -> torch.Tensor:  # pragma: no cover
        """Compute the loss during training given the data.

        Different models can have different outputs and a different number of targets.
        This function has to get overwritten by subclasses.

        Subclasses can use ``_curr_iter`` to write additional information to the tensorboard logs.
        The loss is always written to the logs in ``self.train``.
        """
        raise NotImplementedError

    def write_results(self, results: dict[str, any], prepend: str) -> None:
        """Given a dictionary of results, use the writer to save the values."""
        # pylint: disable=too-many-branches

        for key, value in results.items():
            # regular python value
            if isinstance(value, (int, float, str)):
                self.writer.add_scalar(f"{prepend}/{self.name}/{key}", value, global_step=self.curr_epoch)
            # single valued tensor
            elif isinstance(value, torch.Tensor) and value.ndim == 1 and value.size(0) == 1:
                self.writer.add_scalar(f"{prepend}/{self.name}/{key}", value.item(), global_step=self.curr_epoch)
            # image
            elif isinstance(value, tv_tensors.Image):
                if value.ndim == 3:
                    self.writer.add_image(f"{prepend}/{self.name}/{key}", value, global_step=self.curr_epoch)
                else:
                    self.writer.add_images(f"{prepend}/{self.name}/{key}", value, global_step=self.curr_epoch)
            # Embeddings as dict id -> embed
            elif isinstance(value, tuple) and "_embed" in key:
                ids, embeds = value
                self.writer.add_embedding(
                    embeds, metadata=ids, tag=f"{prepend}/{self.name}/{key}", global_step=self.curr_epoch
                )
            # multiple values as dict sub-key -> sub_value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.writer.add_scalar(
                        f"{prepend}/{self.name}/{key}-{sub_key}", sub_value, global_step=self.curr_epoch
                    )
            # iterable of scalars
            elif isinstance(value, (list, dict, set)):
                for i, sub_value in enumerate(value):
                    self.writer.add_scalar(f"{prepend}/{self.name}/{key}-{i}", sub_value, global_step=self.curr_epoch)
            elif isinstance(value, str):
                self.writer.add_text(tag=key, text_string=value, global_step=self.curr_epoch)
            else:
                warnings.warn(f"Unknown result for writer: {value} {key}, step: {self.curr_epoch}")
        self.logger.debug("results have been written to writer")
        self.writer.flush()

    def print_results(self, results: dict[str, any]) -> None:
        """Given a dictionary of results, print them to the console if allowed."""
        show_images = False
        self.logger.info(f"#### Results - Epoch {self.curr_epoch} ####")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{key}: {value:.2%}")
            elif key.lower() == "cmc":
                self.logger.info("CMC curve:")
                for r, cmc_i in value.items():
                    self.logger.info(f"Rank-{r}: {cmc_i:.1%}")
            elif isinstance(value, tv_tensors.Image):
                show_images = True
                torch_show_image(value, show=False)
            elif "embed" in key:
                continue
            elif isinstance(value, str):
                self.logger.info(f"{key} {value}")
            else:
                warnings.warn(f"Unknown result for printing: {key} {value}")
        # if there were images drawn, show them
        if show_images:
            plt.show()
