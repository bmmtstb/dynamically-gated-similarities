"""
Base model class as lowest building block for dynamic modules
"""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from functools import wraps

import torch
from torch.nn import Module

from dgs.utils.config import get_sub_config
from dgs.utils.constants import PRINT_PRIORITY
from dgs.utils.exceptions import InvalidParameterException, ValidationException
from dgs.utils.types import Config, NodePath, Validations
from dgs.utils.validation import validate_value

module_validations: Validations = {
    "device": [("any", [("in", ["cuda", "cpu"]), ("instance", torch.device)])],
    "is_training": [bool],
    "name": [str, ("longer", 2)],
    "print_prio": [("in", PRINT_PRIORITY)],
    "sp": [bool],
    # optional
    "description": ["optional", str],
    "gpus": [
        "optional",
        list,
        (
            "any",
            [
                ("forall", [int, ("gte", -1)]),
                ("all", [("forall", str), lambda x, _: all(int(x_i) >= -1 for x_i in x)]),
            ],
        ),
    ],
    "log_dir": ["optional", str],
    "num_workers": ["optional", int, ("gte", 0)],
}


def enable_keyboard_interrupt(func: callable) -> callable:  # pragma: no cover
    """Call module.terminate() on Keyboard Interruption (e.g., ctrl+c), which makes sure that all threads are stopped.

    Args:
        func: The decorated function

    Returns:
        Decorated function, which will have advanced keyboard interruption.
    """

    @wraps(func)  # pass information to sphinx through the decorator / wrapper
    def module_wrapper(cls, *args, **kwargs):
        try:
            func(cls, *args, **kwargs)
        except (KeyboardInterrupt, InterruptedError) as e:
            if callable(getattr(cls, "terminate", None)):
                cls.terminate()
            else:
                raise NotImplementedError(f"Class or function {cls} does not have a terminate method.") from e

    return module_wrapper


class BaseModule(ABC):
    r"""Base class for all custom modules.

    Description
    -----------

    Every Module is a building block that can be replaced with other building blocks.
    This defines a base module all of those building blocks inherit.
    This class should not be called directly and should only be inherited by other classes.

    Every module has access to the global configuration for parameters like the modules' device(s).
    Additionally, every module will have own parameters (params) which are a sub node of the overall configuration.

    Configuration
    -------------

    device: (Device, optional, default="cuda")
        The device to run this module and tracker on.

    gpus: (list[int], optional, default=[0])
        List of GPU IDs to use during multi-GPU training or running.

    num_workers: (int, optional, default=0)
        The number of additional workers, the torch DataLoader should use to load the datasets.

    print_prio: (str, optional, default: "INFO")
        How much information should be printed while running.
        Default "INFO" will print status reports but no debugging information.

    sp: (bool, optional, default=True)
        Whether to use a single process (sp) or use multiprocessing.
        If sp is false, 'gpus' has to be defined.

    is_training: (bool, optional, default=False)
        Whether the torch modules should train or evaluate.

    log_dir (FilePath, optional, default="./results/"):
        Path to directory where all the files of this run are saved.
        The subdirectory that represents today will be added to the log directory ("./YYYYMMDD/").

    Attributes:
        config: The overall configuration of the whole algorithm.
        params: The parameters for this specific module.
        _path: Location of params within config as a node path.

    Args:
        config: The overall configuration of the whole algorithm
        path: Keys of config to the parameters of the current module
            e.g. the parameters for the pose estimator will be located in a pose-estimator subgroup of the config
            those key-based paths may be even deeper, just make sure that only information about this specific model
            is stored in params.
    """

    @enable_keyboard_interrupt
    def __init__(self, config: Config, path: NodePath):
        self.config: Config = config
        self.params: Config = get_sub_config(config, path)
        self._path: NodePath = path

        # gpus might be string
        if not self.config["gpus"]:
            self.config["gpus"] = [-1]
        elif isinstance(self.config["gpus"], str):
            self.config["gpus"] = [int(i) for i in self.config["gpus"].split(",")]

        # set default value of num_workers
        if not self.config["num_workers"]:
            self.config["num_workers"] = 0

        self.validate_params(module_validations, "config")

        # set up (file) logger
        self.logger: logging.Logger = self._init_logger()

    def validate_params(self, validations: Validations, attrib_name: str = "params") -> None:
        """Given per key validations, validate this module's parameters.

        Throws exceptions on invalid or nonexistent params.

        Args:
            attrib_name: name of the attribute to validate,
                should be "params" and only for base class "config"
            validations:
                Dictionary with the name of the parameter as key and a list of validations as value.
                Every validation in this list has to be true for the validation to be successful.

                The value for the validation can have multiple types:
                    - A lambda function or other type of callable
                    - A string as reference to a predefined validation function with one argument
                    - None for existence
                    - A tuple with a string as reference to a predefined validation function
                      with one additional argument
                    - It is possible to write nested validations, but then every nested validation has to be a tuple,
                      or a tuple of tuples.
                      For convenience, there are implementations for "any", "all", "not", "eq", "neq", and "xor".
                      Those can have data which is a tuple containing other tuples or validations,
                      or a single validation.
                    - Lists and other iterables can be validated using "forall" running the given validations for every
                      item in the input.
                      A single validation or a tuple of (nested) validations is accepted as data.

        Example:
            This example is an excerpt of the validation for the BaseModule-configuration.

            >>> validations = {
                "device": [
                        str,
                        ("any", (
                            ("in", ["cuda", "cpu"]),
                            ("instance", torch.device)
                ))
                    ],
                    "print_prio": [("in", PRINT_PRIORITY)],
                    "callable": (lambda value: value == 1),
                }

            And within the class :meth:`__init__` call:

            >>> self.validate_params()

        Raises:
            InvalidParameterException: If one of the parameters is invalid.
            ValidationException: If the validation list is invalid or contains an unknown validation.
        """
        for param_name, list_of_validations in validations.items():
            if len(list_of_validations) == 0:
                raise ValidationException(
                    f"Excepted at least one validation, but {param_name} in module {self.__class__.__name__} has zero."
                )

            # check whether param exists in self and raise error if a non-optional param is missing
            if param_name not in getattr(self, attrib_name):
                if "optional" in list_of_validations:
                    continue  # value is optional and does not exist, skip validation
                raise InvalidParameterException(
                    f"'{param_name}' is expected to be in module '{self.__class__.__name__}'"
                )

            # it is now safe to get the value
            value = getattr(self, attrib_name)[param_name]

            for validation in list_of_validations:
                # no validation required except the existence of the current key
                if validation == "optional":
                    continue

                # case name as string or in tuple with additional values
                if isinstance(validation, str | tuple | type):
                    if isinstance(validation, str | type):  # no additional data, therefore set data to None
                        validation_name, data = validation, None
                    else:
                        validation_name, data = validation
                    # call predefined validate
                    if validate_value(value=value, data=data, validation=validation_name):
                        continue
                    raise InvalidParameterException(
                        f"In module '{self.__class__.__name__}', parameter '{param_name}' is not valid. "
                        f"Value is '{value}' and is expected to have validation(s) '{list_of_validations}'."
                    )
                # case custom callable
                if callable(validation):
                    if validation(value):
                        continue
                    raise InvalidParameterException(
                        f"In module {self.__class__.__name__}, parameter {param_name} is not valid. "
                        f"Used a custom validation: {inspect.getsource(validation)}"
                    )

                # no other case was true
                raise ValidationException(
                    f"Validation is expected to be callable or tuple, but is '{type(validation)}'. "
                    f"Current module: '{self.__class__.__name__}', Parameter: '{param_name}'"
                )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> any:  # pragma: no cover
        raise NotImplementedError

    def _init_logger(self) -> logging.Logger:
        """Initialize a basic logger for this module."""
        logger = logging.getLogger(self.name.replace(" ", "."))

        if logger.hasHandlers():
            return logger

        # set level
        prio = self.config.get("print_prio", "INFO")
        log_level = PRINT_PRIORITY[prio] if isinstance(prio, str) else prio
        logger.setLevel(log_level)

        # file handler
        file_handler = logging.FileHandler(
            os.path.join(self.config.get("log_dir", "./results/"), f"output-{self.name}.txt")
        )
        logger.addHandler(file_handler)
        # stdout / stderr handler
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

        # set output format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        return logger

    @property
    def device(self) -> torch.device:
        """Shorthand for getting the device configuration."""
        return torch.device(self.config["device"])

    @property
    def name(self) -> str:
        """Shorthand for getting the name of the module."""
        return self.config["name"]

    def configure_torch_module(self, module: Module, train: bool = None) -> Module:
        """Set compute mode and send model to the device or multiple parallel devices if applicable.

        Args:
            module: The torch module instance to configure.
            train: Whether to train or eval this module, defaults to the value set in the base config.

        Returns:
            The module on the specified device or in parallel.
        """
        train: bool = self.config["is_training"] if train is None else train
        # set torch mode
        if train:
            module.train()
        else:
            module.eval()
        # send model to device(s)
        if (not self.config["sp"] and len(self.config["gpus"]) > 1) or len(self.config["gpus"]) > 1:  # pragma: no cover
            raise NotImplementedError("Parallel does not work yet.")
            # return DistributedDataParallel(module, device_ids=self.config.gpus).to(self.device)
        module.to(device=self.device)
        return module

    def terminate(self) -> None:
        """Terminate this module and all of its submodules.

        If nothing has to be done, just pass.
        Is used for terminating parallel execution and threads in specific models.
        """
