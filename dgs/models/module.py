"""
Base model class as lowest building block for dynamic modules
"""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from datetime import date
from functools import wraps

import torch as t
from torch.nn import Module

from dgs.utils.config import DEF_VAL, get_sub_config
from dgs.utils.constants import PRECISION_MAP, PRINT_PRIORITY, PROJECT_ROOT
from dgs.utils.exceptions import InvalidParameterException, ValidationException
from dgs.utils.files import mkdir_if_missing
from dgs.utils.types import Config, FilePath, NodePath, Validations
from dgs.utils.validation import validate_value

module_validations: Validations = {
    "device": [("any", [("in", ["cuda", "cpu"]), ("instance", t.device), ("startswith", "cuda:")])],
    "is_training": [bool],
    "name": [str, ("longer", 2)],
    # optional
    "print_prio": ["optional", str, ("in", PRINT_PRIORITY)],
    "description": ["optional", str],
    "log_dir": ["optional", str],
    "log_dir_add_date": ["optional", bool],
    "precision": ["optional", ("any", [type, ("in", PRECISION_MAP.keys()), t.dtype])],
}


def enable_keyboard_interrupt(func: callable) -> callable:  # pragma: no cover
    """Call :func:`BaseModule.terminate` on Keyboard Interruption (e.g., ctrl+c),
    which should make sure that all threads are stopped and the GPU memory is freed.

    Args:
        func: The decorated function

    Returns:
        Decorated function, which will have advanced keyboard interruption.
    """

    @wraps(func)  # pass information to sphinx through the decorator / wrapper
    def module_wrapper(cls, *args, **kwargs):
        try:
            return func(cls, *args, **kwargs)
        except (KeyboardInterrupt, InterruptedError) as e:
            if callable(getattr(cls, "terminate", None)):
                return cls.terminate()
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

    device: (Device)
        The torch device to run this module and tracker on.
    is_training: (bool)
        Whether the general torch modules should train or evaluate.
        Modes of different modules can be set individually using '.eval()', '.train()', or the functions from
        :mod:`.dgs.utils.torchtools`.
    name (str):
        The name of this configuration.
        Mostly used for printing, logging, and file saving.

    Optional Configuration
    ----------------------

    print_prio: (str, optional)
        How much information should be printed while running.
        "INFO" will print status reports but no debugging information.
        Default: ``DEF_VAL.base.print_prio`` .
    description (str, optional):
        The description of the overall configuration.
        Default: ``DEF_VAL.base.description`` .
    log_dir (FilePath, optional):
        Path to directory where all the files of this run are saved.
        The date will be added to the path if ``log_dir_add_date`` is ``True``.
        Default: ``DEF_VAL.base.log_dir`` .
    log_dir_add_date (bool, optional):
        Whether to append the date to the ``log_dir``.
        If ``True``, The subdirectory that represents today will be added to the log directory ("./YYYYMMDD/").
        Default: ``DEF_VAL.base.log_dir_add_date`` .
    precision (Union[type, str, torch.dtype], optional)
        The precision at which this module should operate.
        Default: ``DEF_VAL.base.precision`` .

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

        self.validate_params(module_validations, "config")

        # set up (file) logger

        self.log_dir: FilePath = os.path.normpath(
            os.path.abspath(
                os.path.join(
                    PROJECT_ROOT,
                    self.config.get("log_dir", DEF_VAL["base"]["log_dir"]),
                    (
                        f"./{date.today().strftime('%Y%m%d')}/"
                        if self.config.get("log_dir_add_date", DEF_VAL["base"]["log_dir_add_date"])
                        else ""
                    ),
                )
            )
        )
        mkdir_if_missing(self.log_dir)
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
                        ("any",
                            [
                                ("in", ["cuda", "cpu"]),
                                ("instance", torch.device)
                            ]
                        )
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
                if isinstance(validation, (str, tuple, type)):
                    if isinstance(validation, (str, type)):  # no additional data, therefore set data to None
                        validation_name, data = validation, None
                    else:
                        validation_name, data = validation
                    # call predefined validate
                    if validate_value(value=value, data=data, validation=validation_name):
                        continue
                    raise InvalidParameterException(
                        f"In module '{self.__class__.__name__}', parameter '{param_name}' is not valid. "
                        f"Value is '{value}' and is expected to have validation '{validation_name}' with data '{data}'."
                        f"\n\nTotal list of validations: '{list_of_validations}'."
                    )
                # case custom callable
                if callable(validation):
                    validation: callable
                    if validation(value):
                        continue
                    raise InvalidParameterException(
                        f"In module {self.__class__.__name__}, parameter '{param_name}' is not valid. "
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
        logger = logging.getLogger(self.name_safe)

        if logger.hasHandlers():
            return logger

        # set level
        prio = self.config.get("print_prio", DEF_VAL["base"]["print_prio"])
        log_level = PRINT_PRIORITY[prio] if isinstance(prio, str) else prio
        logger.setLevel(log_level)

        # file handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f"output-{self.name_safe}.txt"), delay=True)
        logger.addHandler(file_handler)
        # stdout / stderr handler
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

        # set output format
        formatter = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        return logger

    @property
    def is_training(self) -> bool:
        """Get whether this module is set to training-mode."""
        return self.config["is_training"]

    @property
    def device(self) -> t.device:
        """Get the device of this module."""
        return t.device(self.config["device"])

    @property
    def name(self) -> str:
        """Get the name of the module."""
        return str(self.config["name"])

    @property
    def name_safe(self) -> str:
        """Get the escaped name of the module usable in filepaths by replacing spaces and underscores."""
        return str(self.config["name"]).replace(" ", "-").replace(".", "_")

    @property
    def precision(self) -> t.dtype:
        """Get the (floating point) precision used in multiple parts of this module."""
        precision = self.config.get("precision", DEF_VAL["base"]["precision"])
        if isinstance(precision, t.dtype):
            return precision
        if precision == int:
            return t.int
        if precision == float:
            return t.float
        if isinstance(precision, str):
            return PRECISION_MAP[precision]
        raise NotImplementedError

    def configure_torch_module(self, module: Module, train: bool = None) -> Module:
        """Set compute mode and send model to the device or multiple parallel devices if applicable.

        Args:
            module: The torch module instance to configure.
            train: Whether to train or eval this module, defaults to the value set in the base config.

        Returns:
            The module on the specified device or in parallel.
        """
        train: bool = self.is_training if train is None else train
        # set torch mode
        if train:
            module.train()
        else:
            module.eval()
        # send model to device(s) - multiple devices not supported
        module = module.to(device=self.device)
        return module

    def terminate(self) -> None:  # pragma: no cover
        """Terminate this module and all of its submodules.

        If nothing has to be done, just pass.
        Is used for terminating parallel execution and threads in specific models.
        """
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)
        del self.logger
        t.cuda.empty_cache()
