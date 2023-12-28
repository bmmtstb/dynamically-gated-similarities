"""
Base model class as lowest building block for dynamic modules
"""
import inspect
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
    "name": ["str", ("longer", 2)],
    "batch_size": ["int", ("gte", 1)],
    "print_prio": [("in", PRINT_PRIORITY)],
    "device": ["str", ("or", (("in", ["cuda", "cpu"]), ("instance", torch.device)))],
    "gpus": [lambda gpus: isinstance(gpus, list) and all(isinstance(gpu, int) for gpu in gpus)],
    "num_workers": ["int", ("gte", 0)],
    "sp": [("instance", bool)],
    "training": [("instance", bool)],
}


def enable_keyboard_interrupt(func: callable) -> callable:
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
        except KeyboardInterrupt as e:
            if callable(getattr(cls, "terminate", None)):
                cls.terminate()
            else:
                raise NotImplementedError(f"Class or function {cls} does not have a terminate method.") from e

    return module_wrapper


def configure_torch_module(orig_cls):
    """Decorator to decorate classes, which have to be a torch.nn.Module,
    to call BaseModule.configure_torch_model on themselves.

    :param orig_cls: The decorated class.
    :return: The decorated class after the configuration is applied.
    """
    orig_init = orig_cls.__init__

    def class_wrapper(self, *args, **kwargs):
        if not isinstance(self, BaseModule) or not isinstance(self, Module):
            raise NotImplementedError(
                f"Given class or function {self} is not a child of BaseModule and torch.nn.Module"
            )
        # first initialize class
        orig_init(self, *args, **kwargs)
        # then call configure_torch_model()
        self.configure_torch_model(module=self)

    # override original init method
    orig_cls.__init__ = class_wrapper
    return orig_cls


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

    batch_size: (int, optional, default: 32)
        The batch size of the tracker, also used in all other modules, if not specified otherwise.

    device: (Device, optional, default="cuda")
        The device to run this module and tracker on.

    gpus: (list[int], optional, default=[0])
        List of GPU IDs to use during multi-GPU training or running.

    num_workers: (int, optional, default=0)
        The number of additional workers, the torch DataLoader should use to load the datasets.

    print_prio: (str, optional, default: "normal")
        How much information should be printed while running.

    sp: (bool, optional, default=True)
        Whether to use a single process (sp) or use multiprocessing.
        If sp is false, 'gpus' has to be defined.

    training: (bool, optional, default=False)
        Whether the torch modules should train or evaluate.

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
            self.config["gpus"] = (
                [int(i) for i in self.config["gpus"].split(",")] if torch.cuda.device_count() >= 1 else [-1]
            )

        # validate config when calling BaseModule class and not when calling its children
        if self.__class__.__name__ == "BaseModule":  # fixme always true, even for child modules
            self.validate_params(module_validations, "config")

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
                    - a lambda function or other type of callable
                    - a string as reference to a predefined validation function with one argument
                    - None for existence
                    - a tuple with a string as reference to a predefined validation function
                        with one additional argument
                    - it is possible to write nested validations, but then every nested validation has to be a tuple,
                        or a tuple of tuples.
                        For convenience "or" and "and" can have unlimited tuples as their second argument, therefore
                        acting as replacements for "any" and "all".

        Example:
            This example is an excerpt of the validation of the BaseModule-configuration.

            >>> validations = {
                "device": [
                        "str",
                        ("or", (
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
            InvalidParameterException: If one of the parameters is invalid
            ValueError: If the argument validation has an unknown type

        """
        for param_name, list_of_validations in validations.items():
            if len(list_of_validations) == 0:
                raise ValidationException(
                    f"Excepted at least one validation, but {param_name} in module {self.__class__.__name__} has zero."
                )

            for validation in list_of_validations:
                # check whether param exists in self
                if param_name not in getattr(self, attrib_name):
                    raise InvalidParameterException(
                        f"{param_name} is expected to be in module {self.__class__.__name__}"
                    )

                if validation is None:
                    # no validation required except the existence of the current key
                    continue

                value = getattr(self, attrib_name)[param_name]

                # case custom callable
                if callable(validation):
                    if validation(value):
                        continue
                    raise InvalidParameterException(
                        f"In module {self.__class__.__name__}, parameter {param_name} is not valid. "
                        f"Used a custom validation: {inspect.getsource(validation)}"
                    )

                # case name as string or in tuple with additional values
                if isinstance(validation, str | tuple):
                    if isinstance(validation, str):  # no additional data, therefore set data to None
                        validation_name, data = validation, None
                    else:
                        validation_name, data = validation
                    # call predefined validate
                    if validate_value(value=value, data=data, validation=validation_name):
                        continue
                    raise InvalidParameterException(
                        f"In module {self.__class__.__name__}, parameter {param_name} is not valid. "
                        f"Value is {value} and is expected to have validation {validation_name}."
                    )
                # no other case was true
                raise ValidationException(
                    f"Validation is expected to be callable or tuple, but is {type(validation)}. "
                    f"Current module: {self.__class__.__name__}, Parameter: {param_name}"
                )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> any:
        raise NotImplementedError

    def print(self, priority: str) -> bool:
        """Check whether the Module is allowed to print something with the given priority.

        Args:
            priority: Priority on which this will print.
                Value has to be in PRINT_PRIO.
                But this is kind of counterintuitive:
                - Use 'normal' if you want to print it all the time as long as cfg.print_prio is not 'none'
                - Use 'debug' if you want to print it iff cfg.print_prio is either 'debug' or 'all'
                - Use 'all' if you want to print it iff cfg.print_prio == 'all'



        Returns:
            Whether the module is allowed to print given its priority.
        """
        try:
            index_given: int = PRINT_PRIORITY.index(priority)
        except ValueError as verr:
            raise ValueError(f"Priority: {priority} is not in {PRINT_PRIORITY}") from verr
        if priority == "none":
            raise ValueError("To print with priority of none doesn't make sense...")

        index_current: int = PRINT_PRIORITY.index(self.config["print_prio"])

        return index_given <= index_current

    @property
    def device(self) -> torch.device:
        """Shorthand for getting the device configuration."""
        return torch.device(self.config["device"])

    def configure_torch_model(self, module: Module, train: bool = None) -> Module:
        """Set compute mode and send model to the device or multiple parallel devices if applicable.

        Args:
            module: The torch module instance to configure.
            train: Whether to train or eval this module, defaults to the value set in the base config.

        Returns:
            The module on the specified device or in parallel.
        """
        train: bool = self.config["training"] if train is None else train
        # set torch mode
        if train:
            module.train()
        else:
            module.eval()
        # send model to device(s)
        if (not self.config["sp"] and len(self.config["gpus"]) > 1) or len(self.config["gpus"]) > 1:
            raise NotImplementedError("Parallel does not work yet.")
            # return DistributedDataParallel(module, device_ids=self.config.gpus).to(self.device)
        module.to(device=self.device)
        return module

    def terminate(self) -> None:
        """Terminate this module and all of its submodules.

        If nothing has to be done, just pass.
        Is used for terminating parallel execution and threads in specific models.
        """
