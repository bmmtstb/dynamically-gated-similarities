"""
Class and functions used during training and testing of different modules.
"""

import logging
import os
import time
import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Union

import torch as t
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import tv_tensors as tvte
from tqdm import tqdm

from dgs.models.loader import module_loader
from dgs.models.loss import get_loss_function, LOSS_FUNCTIONS
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.modules.named import NamedModule
from dgs.models.optimizer import get_optimizer, OPTIMIZERS
from dgs.models.scheduler import get_scheduler, SCHEDULERS
from dgs.utils.config import DEF_VAL, get_sub_config, save_config
from dgs.utils.constants import MODULE_TYPES
from dgs.utils.exceptions import InvalidConfigException
from dgs.utils.state import State
from dgs.utils.timer import DifferenceTimer, DifferenceTimers
from dgs.utils.torchtools import resume_from_checkpoint, save_checkpoint
from dgs.utils.types import Config, FilePath, NodePath, Results, Validations
from dgs.utils.visualization import torch_show_image

engine_validations: Validations = {
    "model_path": ["NodePath"],
    "model_type": [str, ("in", MODULE_TYPES)],
    # optional
    "writer_kwargs": ["optional", dict],
    "writer_log_dir_suffix": ["optional", str],
}

train_validations: Validations = {
    "loss": [("any", ["callable", ("in", LOSS_FUNCTIONS)])],
    "optimizer": [("any", ["callable", ("in", OPTIMIZERS)])],
    # optional
    "epochs": ["optional", int, ("gte", 1)],
    "loss_kwargs": ["optional", dict],
    "optimizer_kwargs": ["optional", dict],
    "save_interval": ["optional", int, ("gte", 1)],
    "scheduler": ["optional", ("any", ["callable", ("in", SCHEDULERS)])],
    "scheduler_kwargs": ["optional", dict],
    "train_load_image_crops": ["optional", bool],
}

test_validations: Validations = {
    # optional
    "normalize": ["optional", bool],
}


class EngineModule(NamedModule, nn.Module):
    """Module for training, validating, and testing other Modules.

    Most of the settings are defined within the configuration file in the `training` section.

    Params
    ------

    model_path (NodePath):
        Path to the configuration setting up the model to be trained or tested.

    model_type (str):
        The type of :class:`.BaseModule` to be loaded as the model for training and testing.
        The value will be passed as ``module_type`` in the :func:`module_loader` call.

    Optional Params
    ---------------

    writer_kwargs (dict, optional):
        Additional kwargs for the torch writer.
        Default ``DEF_VAL.engine.test.writer_kwargs``.
    writer_log_dir_suffix (str, optional):
        Additional subdirectory or name suffix for the torch writer.
        Default ``DEF_VAL.engine.test.writer_log_dir_suffix``.


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

    normalize (bool, optional):
        Whether to normalize the prediction and target during testing.
        Default ``DEF_VAL.engine.test.normalize``.

    Optional Train Params
    ---------------------

    epochs (int, optional):
        The number of epochs to run the training for.
        Default ``DEF_VAL.engine.train.epochs``.
    optimizer_kwargs (dict, optional):
        Additional kwargs for the optimizer.
        Default ``DEF_VAL.engine.train.optim_kwargs``.
    scheduler (str|callable, optional):
        The name or instance of a scheduler.
        If you want to use different or multiple schedulers, you can chain them using
        ``torch.optim.lr_scheduler.ChainedScheduler`` or create a custom Scheduler and register it.
        Default ``DEF_VAL.engine.train.scheduler``.
    scheduler_kwargs (dict, optional):
        Additional kwargs for the scheduler.
        Keep in mind that the different schedulers need fairly different kwargs.
        The optimizer will be passed to the scheduler during initialization as the `optimizer` keyword argument.
        Default ``DEF_VAL.engine.train.scheduler_kwargs``.
    loss_kwargs (dict, optional):
        Additional kwargs for the loss.
        Default ``DEF_VAL.engine.train.loss_kwargs``.
    save_interval (int, optional):
        The interval for saving (and evaluating) the model during training.
        Default ``DEF_VAL.engine.train.save_interval``.
    start_epoch (int, optional):
        The epoch at which to start.
        (In the end the epochs are 1-indexed, but it shouldn't matter as long as you stick with one format.
        Default ``DEF_VAL.engine.train.start_epoch``.
    train_load_image_crops (bool, optional):
        Whether to load the image crops during training.
        Default ``DEF_VAL.engine.train.load_image_crops``.
    """

    # The engine is the heart of most algorithms and therefore contains a los of stuff.
    # pylint: disable = too-many-instance-attributes, too-many-arguments

    loss: nn.Module
    model: nn.Module
    writer: SummaryWriter

    curr_epoch: int = 0

    test_dl: TorchDataLoader
    """The torch DataLoader containing the test data."""

    val_dl: TorchDataLoader
    """The torch DataLoader containing the validation data."""

    train_dl: TorchDataLoader
    """The torch DataLoader containing the training data."""

    train_load_image_crops: bool = True
    """Whether to load the image crops during training."""

    def __init__(
        self,
        config: Config,
        path: NodePath,
        test_loader: TorchDataLoader,
        *,
        val_loader: TorchDataLoader = None,
        train_loader: TorchDataLoader = None,
        **_kwargs,
    ):
        NamedModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        # ###### #
        # PARAMS #
        # ###### #
        self.validate_params(engine_validations)
        # logging
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.log_dir,
                self.params.get("writer_log_dir_suffix", DEF_VAL["engine"]["writer_log_dir_suffix"]),
            ),
            comment=self.config.get("description"),
            **self.params.get("writer_kwargs", DEF_VAL["engine"]["writer_kwargs"]),
        )
        # Set up DGSModule
        model = module_loader(config=config, module_type=self.params["model_type"], key=self.params["model_path"])
        self.register_module("model", self.configure_torch_module(model))
        # save config in the out-folder to make sure values haven't changed when validating those files
        save_config(
            filepath=os.path.join(self.log_dir, f"config-{self.name_safe}-{datetime.now().strftime('%Y%m%d_%H_%M')}"),
            config=config,
        )
        # save default values
        save_config(
            filepath=os.path.join(self.log_dir, f"default-values-{datetime.now().strftime('%Y%m%d_%H_%M')}"),
            config=DEF_VAL,
        )

        # #### #
        # TEST #
        # #### #
        self.params_test: Config = get_sub_config(config, ["test"])
        self.validate_params(test_validations, attrib_name="params_test")
        self.test_dl = test_loader

        # ##### #
        # TRAIN #
        # ##### #
        self.params_train: Config = {}
        if self.is_training:
            if "train" not in config:
                raise KeyError("'is_training' is True, but there is no key in the config named 'train'")
            self.params_train = get_sub_config(config, ["train"])
            self.validate_params(train_validations, attrib_name="params_train")
            if train_loader is None:
                raise InvalidConfigException("is_training is turned on but train_loader is None.")
            if val_loader is None:
                raise InvalidConfigException("is_training is turned on but val_loader is None.")

            # save train and validation data loader
            self.train_dl = train_loader
            self.val_dl = val_loader

            # epochs
            self.epochs: int = self.params_train.get("epochs", DEF_VAL["engine"]["train"]["epochs"])
            self.start_epoch: int = self.params_train.get("start_epoch", DEF_VAL["engine"]["train"]["start_epoch"])
            self.curr_epoch = self.start_epoch
            self.save_interval: int = self.params_train.get(
                "save_interval", DEF_VAL["engine"]["train"]["save_interval"]
            )

            # modules
            self.loss = get_loss_function(self.params_train["loss"])(
                **self.params_train.get(
                    "loss_kwargs", DEF_VAL["engine"]["train"]["loss_kwargs"]
                )  # optional loss kwargs
            )
            self.optimizer = None
            self.lr_sched = None

            # params
            self.train_load_image_crops = self.params_train.get(
                "load_image_crops", DEF_VAL["engine"]["train"]["load_image_crops"]
            )

    @property
    def module_type(self) -> str:
        return "engine"

    def initialize_optimizer(self) -> None:
        """Because the module might be set after the initial step,
        load the optimizer and scheduler at the start of the training.
        """
        self.optimizer = get_optimizer(self.params_train["optimizer"])(
            self.model.parameters(),
            **self.params_train.get("optimizer_kwargs", DEF_VAL["engine"]["train"]["optim_kwargs"]),
        )
        # the learning-rate schedulers need the optimizer for instantiation
        self.lr_sched = get_scheduler(self.params_train.get("scheduler", DEF_VAL["engine"]["train"]["scheduler"]))(
            optimizer=self.optimizer,
            **self.params_train.get("scheduler_kwargs", DEF_VAL["engine"]["train"]["scheduler_kwargs"]),
        )

    @enable_keyboard_interrupt
    def __call__(self, *args, **kwargs) -> any:
        return self.run(*args, **kwargs)

    @abstractmethod
    def get_data(self, ds: State) -> any:
        """Function to retrieve the data used in the model's prediction from the train- and test- DataLoaders."""
        raise NotImplementedError

    @abstractmethod
    def get_target(self, ds: State) -> any:
        """Function to retrieve the evaluation targets from the train- and test- DataLoaders."""
        raise NotImplementedError

    @enable_keyboard_interrupt
    def run(self) -> None:
        """Run the model. First train, then test!"""
        self.logger.info(f"#### Starting run {self.name} ####")
        if "description" in self.config:
            self.logger.info(f"Config Description: {self.config['description']}")

        if self.is_training:
            # train and eval the model
            self.train_model()

        self.test()

    @abstractmethod
    @enable_keyboard_interrupt
    def evaluate(self) -> Results:
        """Run tests, defined in Sub-Engine.

        Returns:
            dict[str, any]: A dictionary containing all the computed accuracies, metrics, ... .
        """
        raise NotImplementedError

    @abstractmethod
    @enable_keyboard_interrupt
    def test(self) -> Results:
        """Run tests, defined in Sub-Engine.

        Returns:
            dict[str, any]: A dictionary containing all the computed accuracies, metrics, ... .
        """
        raise NotImplementedError

    @abstractmethod
    @enable_keyboard_interrupt
    def predict(self) -> any:
        """Given test data, predict the results without evaluation.

        Returns:
            The predicted results. Datatype might vary depending on the used engine.
        """
        raise NotImplementedError

    @enable_keyboard_interrupt
    def train_model(self) -> optim.Optimizer:
        """Train the given model using the given loss function, optimizer, and learning-rate schedulers.

        After every epoch, the current model is tested and the current model is saved.

        Returns:
            The current optimizer after training.
        """
        # pylint: disable=too-many-statements,too-many-branches

        if self.train_dl is None:
            raise ValueError("No DataLoader for the Training data was given. Can't continue.")
        if (
            self.model is None
            or not isinstance(self.model, nn.Module)
            or (isinstance(self.model, nn.Sequential) and len(self.model) == 0)
        ):
            raise ValueError("No model was given. Can't continue.")
        if (
            not hasattr(self, "optimizer")
            or self.optimizer is None
            or not hasattr(self, "lr_sched")
            or self.lr_sched is None
        ):
            try:
                self.initialize_optimizer()
            except Exception as e:
                raise ValueError("No optimizer or lr scheduler given. Or Failed to initialize. Can't continue.") from e

        self.logger.debug("#### Start Training ####")

        # initialize variables
        timers: DifferenceTimers = DifferenceTimers()
        epoch_t: DifferenceTimer = DifferenceTimer()
        data: Union[State, list[State]]

        for self.curr_epoch in (
            pbar_epoch := tqdm(range(self.start_epoch, self.epochs + 1), desc="Train - Epoch", position=0)
        ):
            pbar_epoch.set_postfix_str(str(self.curr_epoch))
            self.logger.debug(f"#### Training - Epoch {self.curr_epoch} ####")

            # set model mode to train, and make sure to do it here, because the evaluation script will change it
            self.set_model_mode("train")

            self.model.zero_grad()  # fixme, is this required?
            self.optimizer.zero_grad()

            epoch_loss = 0
            time_epoch_start = time.time()
            time_batch_start = time.time()  # reset timer for retrieving the data

            # loop over all the data
            for batch_idx, data in (
                pbar_batch := tqdm(
                    enumerate(self.train_dl),
                    desc="Train - Batch",
                    position=1,
                    total=len(self.train_dl),
                    leave=False,
                )
            ):
                pbar_batch.set_postfix_str(str(batch_idx))

                # load image crops if required
                if self.train_load_image_crops:
                    if isinstance(data, list):
                        for d in data:
                            d.load_image_crop(store=True)
                    elif isinstance(data, State):
                        data.load_image_crop(store=True)

                curr_iter = (self.curr_epoch - 1) * len(self.train_dl) + batch_idx
                timers.add(name="data", prev_time=time_batch_start)

                time_optim_start = time.time()

                # OPTIMIZE MODEL
                with t.enable_grad():
                    self.optimizer.zero_grad()
                    self.model.zero_grad()
                    loss = self._get_train_loss(data, curr_iter)
                    loss.backward()
                    self.optimizer.step()
                # OPTIMIZE END

                timers.add(name="forwbackw", prev_time=time_optim_start)
                batch_t = timers.add(name="batch", prev_time=time_batch_start)

                epoch_loss += loss.item()
                self.writer.add_scalars(
                    main_tag="Train/loss",
                    tag_scalar_dict={"curr": loss.item(), "avg": epoch_loss / float(batch_idx + 1)},
                    global_step=curr_iter,
                )
                self.writer.add_scalars(
                    main_tag="Train/time",
                    tag_scalar_dict={"indiv": batch_t / len(data), **timers.get_last()},
                    global_step=curr_iter,
                )
                self.writer.add_scalar("Train/lr", self.optimizer.param_groups[-1]["lr"], global_step=curr_iter)

                # clean or remove all the image tensors to free up cuda memory
                if isinstance(data, State):
                    data.clean()
                elif isinstance(data, list):
                    for d in data:
                        d.clean()

                # ############ #
                # END OF BATCH #
                # ############ #
                time_batch_start = time.time()  # reset timer for retrieving the data before entering next loop

            # ############ #
            # END OF EPOCH #
            # ############ #
            epoch_t.add(time_epoch_start)

            # write and log the loss
            self.writer.add_hparams(
                run_name=self.name_safe,
                hparam_dict={"curr_lr": self.optimizer.param_groups[-1]["lr"], **self.get_hparam_dict()},
                metric_dict={
                    "epoch_loss": epoch_loss,
                    **{f"time_avg_{name}": val for name, val in timers.get_avgs().items()},
                    **{f"time_sum_{name}": val for name, val in timers.get_sums().items()},
                },
                global_step=self.curr_epoch,
            )
            self.writer.add_scalar(tag="Train/epoch_loss", scalar_value=epoch_loss, global_step=self.curr_epoch)
            self.logger.info(epoch_t.print(name="epoch", prepend="Training", hms=True))

            if self.curr_epoch % self.save_interval == 0:
                # evaluate current model every few epochs
                metrics: dict[str, any] = self.evaluate()
                self.save_model(
                    epoch=self.curr_epoch, metrics=metrics, optimizer=self.optimizer, lr_sched=self.lr_sched
                )

                if len(metrics) > 0:
                    self.writer.add_hparams(
                        run_name=self.name_safe,
                        hparam_dict={"curr_lr": self.optimizer.param_groups[-1]["lr"], **self.get_hparam_dict()},
                        metric_dict=metrics,
                        global_step=self.curr_epoch,
                    )

            # handle updating the learning rate scheduler
            self.lr_sched.step()

            # update and force write the writer
            self.writer.flush()

        # ############### #
        # END OF TRAINING #
        # ############### #

        self.logger.info(epoch_t.print(name=f"epoch {self.curr_epoch}", prepend="Training", hms=True))
        self.logger.info("#### Training complete ####")

        self.writer.flush()

        return self.optimizer

    def set_model_mode(self, mode: str) -> None:
        """Set model mode to train or test."""
        if mode not in ["train", "test", "eval"]:
            raise ValueError(f"unknown mode: {mode}")
        # set model to train mode
        if not hasattr(self.model, mode):
            warnings.warn(f"`model.{mode}()` is not available.")
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

    def save_model(
        self, epoch: int, metrics: dict[str, any], optimizer: optim.Optimizer, lr_sched: optim.lr_scheduler.LRScheduler
    ) -> None:  # pragma: no cover
        """Save the current model and other weights into a '.pth' file.

        Args:
            epoch: The epoch this model is saved.
            metrics: A dict containing the computed metrics for this module.
            optimizer: The current optimizer
            lr_sched: The current learning rate scheduler.
        """
        curr_lr = f"{optimizer.param_groups[-1]['lr']:.10f}".replace(".", "_")

        save_checkpoint(
            state={
                "model": self.model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_sched.state_dict(),
            },
            save_dir=os.path.join(self.log_dir, "./checkpoints/"),
            prepend=f"lr{curr_lr}",
            verbose=self.logger.isEnabledFor(logging.INFO),
        )

    def load_model(self, path: FilePath) -> None:  # pragma: no cover
        """Load the model from a file. Set the start epoch to the epoch + 1 of the specified in the loaded model.

        Notes:
            Loads the states of the ``optimizer`` and ``lr_scheduler`` if they are present in the engine
            (e.g. during training) and the respective data is given in the checkpoint at the ``path``.

        Args:
            path: The path to the checkpoint where this model was saved.
        """
        last_epoch = resume_from_checkpoint(
            fpath=path,
            model=self.model,
            optimizer=self.optimizer if hasattr(self, "optimizer") else None,
            scheduler=self.lr_sched if hasattr(self, "lr_sched") else None,
            verbose=self.logger.isEnabledFor(logging.DEBUG),
        )
        self.start_epoch = last_epoch + 1
        self.curr_epoch = self.start_epoch

    def terminate(self) -> None:  # pragma: no cover
        """Handle forceful termination, e.g., ctrl+c"""
        if hasattr(self, "writer"):
            self.writer.flush()
            self.writer.close()
        for attr in ["model", "optimizer", "lr_sched", "test_dl", "train_dl", "val_dl", "loss", "module"]:
            if hasattr(self, attr):
                delattr(self, attr)
        t.cuda.empty_cache()
        super().terminate()

    def get_hparam_dict(self) -> dict[str, any]:
        """Get the hyperparameters of the current engine.
        Child-modules can inherit this method and add additional hyperparameters.

        By default, all parameters from test and training are added to the hparam_dict.
        """

        def flatten_dict(parent_dict: dict[str, any], parent_key: str, child_dict: dict[str, any]) -> None:
            """Flatten a nested dictionary in place."""
            for sub_key, sub_value in child_dict.items():
                new_key = f"{parent_key}_{sub_key}"
                if isinstance(sub_value, dict):
                    flatten_dict(parent_dict, parent_key=new_key, child_dict=sub_value)
                else:
                    parent_dict[new_key] = sub_value

        hparams = {
            "base_lr": self.params_train["optimizer_kwargs"]["lr"],
            "batch_size_test": self.test_dl.batch_size if self.test_dl is not None else -1,
            "batch_size_val": self.val_dl.batch_size if self.val_dl is not None else -1,
            "batch_size_train": self.train_dl.batch_size if self.train_dl is not None else -1,
        }

        flatten_dict(parent_dict=hparams, parent_key="test", child_dict=self.params_test)
        flatten_dict(parent_dict=hparams, parent_key="train", child_dict=self.params_train)

        # SummaryWriter - value should be one of int, float, str, bool, or torch.Tensor
        for k, v in hparams.items():
            if isinstance(v, dict):
                flatten_dict(parent_dict=hparams, parent_key=k, child_dict=v)
            if isinstance(v, (tuple, list)):
                try:
                    hparams[k] = t.tensor(v)
                except ValueError:
                    hparams[k] = str(v)

        return hparams

    def _normalize_test(self, tensor: t.Tensor) -> t.Tensor:
        """If ``params_test.normalize`` is True, we want to obtain the normalized prediction and target."""
        if self.params_test.get("normalize", DEF_VAL["engine"]["test"]["normalize"]):
            self.logger.debug("Normalizing test data")
            tensor: t.Tensor = nn.functional.normalize(tensor)
        return tensor

    @abstractmethod
    @t.enable_grad()
    def _get_train_loss(self, data: Union[State, list[State]], _curr_iter: int) -> t.Tensor:  # pragma: no cover
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
                self.writer.add_scalar(f"{prepend}/{key}", value, global_step=self.curr_epoch)
            # single valued tensor
            elif isinstance(value, t.Tensor) and value.ndim == 1 and value.size(0) == 1:
                self.writer.add_scalar(f"{prepend}/{key}", value.item(), global_step=self.curr_epoch)
            # image
            elif isinstance(value, tvte.Image):
                if value.ndim == 3:
                    self.writer.add_image(f"{prepend}/{key}", value, global_step=self.curr_epoch)
                else:
                    self.writer.add_images(f"{prepend}/{key}", value, global_step=self.curr_epoch)
            # Embeddings as dict id -> embed
            elif isinstance(value, tuple) and "_embed" in key:
                ids, embeds = value
                self.writer.add_embedding(embeds, metadata=ids, tag=f"{prepend}/{key}", global_step=self.curr_epoch)
            # multiple values as dict can be written using add_scalars
            elif isinstance(value, dict):
                self.writer.add_scalars(
                    main_tag=f"{prepend}/{key}",
                    tag_scalar_dict={str(k): v for k, v in value.items()},
                    global_step=self.curr_epoch,
                )
            # iterable of scalars
            elif isinstance(value, (list, dict, set)):
                for i, sub_value in enumerate(value):
                    self.writer.add_scalar(f"{prepend}/{key}-{i}", sub_value, global_step=self.curr_epoch)
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
            if isinstance(value, float):
                self.logger.info(f"{key}: {value:.2%}")
            elif isinstance(value, int):
                self.logger.info(f"{key}: {value}")
            elif key.lower() == "cmc":
                self.logger.info("CMC curve:")
                for r, cmc_i in value.items():
                    self.logger.info(f"Rank-{r}: {cmc_i:.2%}")
            elif isinstance(value, tvte.Image):
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
