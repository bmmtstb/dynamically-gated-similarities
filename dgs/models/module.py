"""
Base model class as lowest building block for dynamic modules
"""
import inspect
from abc import ABC, abstractmethod

import torch
from torch.nn import Module

from dgs.utils.config import get_sub_config, validate_value
from dgs.utils.constants import PRINT_PRIORITY
from dgs.utils.exceptions import InvalidParameterException, ValidationException
from dgs.utils.types import Config, Device, NodePath, Validations

module_validations: Validations = {
    "device": ["str", ("or", (("in", ["cuda", "cpu"]), ("instance", torch.device)))],
    "gpus": [lambda gpus: isinstance(gpus, list) and all(isinstance(gpu, int) for gpu in gpus)],
    "print_prio": [("in", PRINT_PRIORITY)],
    "sp": [("instance", bool)],
    "training": [("instance", bool)],
}


def enable_keyboard_interrupt(func: callable) -> callable:
    """Call module.terminate() on Keyboard Interruption (e.g. ctrl+c), which makes sure that all threads are stopped.

    Args:
        func: The decorated function

    Returns:
        Decorated function, which will have advanced keyboard interruption.
    """

    def module_wrapper(cls, *args, **kwargs):
        """wrapper for decorator"""
        try:
            func(cls, *args, **kwargs)
        except KeyboardInterrupt as e:
            if callable(getattr(cls, "terminate", None)):
                cls.terminate()
            else:
                raise NotImplementedError(f"Class or function {cls} does not have a terminate method.") from e

    return module_wrapper


class BaseModule(ABC):
    """
    Every Module is a building block that can be replaced with other building blocks.
    This defines a base module all of those building blocks inherit.
    This class should not be called directly only inherited by other classes.


    Attributes:
        config (Config): The overall configuration of the whole algorithm
        params (Config): The parameters for this specific module
        _path (NodePath): Location of params within config as a node path
    """

    @enable_keyboard_interrupt
    def __init__(self, config: Config, path: NodePath):
        """
        Every module has access the global configuration for parameters like the modules' device(s).
        Additionally, every module will have own parameters (params) which are a sub node of the overall configuration.

        Args:
            config: The overall configuration of the whole algorithm
            path: Keys of config to the parameters of the current module
                e.g. the parameters for the pose estimator will be located in a pose-estimator subgroup of the config
                those key-based paths may be even deeper, just make sure that only information about this specific model
                is stored in params
        """
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
        if self.__class__.__name__ == "BaseModule":
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
            This results in more or less the validation for the config of the BaseModule.

            ::

                validations = {
                    "device": [
                        "str",
                        ("or", (
                            ("in", ["cuda", "cpu"]),
                            ("instance", torch.device)
                        ))
                    ],
                    "print_prio": [("in", PRINT_PRIORITY)],
                    "callable": (lambda value: value == 1)
                }

            And within the class `__init__` call:
            ::

                >> self.validate_params()

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
                - Use "normal" if you want to print it all the time as long as cfg.print_prio is not "none"
                - Use "debug" if you want to print it iff cfg.print_prio is either "debug" or "alL"
                - Use "all" if you want to print it iff cfg.print_prio == "alL"



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
    def device(self) -> Device:
        """Shorthand for get device property."""
        return self.config["device"]

    def configure_torch_model(self, module: Module, train: bool = None) -> Module:
        """Set compute mode and send model to device or parallel devices if applicable.

        Args:
            module: The torch module instance to configure.
            train: Whether to train or eval this module, defaults to the value set in config.

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
        if not self.config["sp"] and len(self.config.gpus) > 1:
            raise NotImplementedError("Parallel does not work yet.")
            # return DistributedDataParallel(module, device_ids=self.config.gpus).to(self.device)
        module.to(self.device)
        return module

    def terminate(self) -> None:
        """Terminate this module and all of its submodules.

        If nothing has to be done, just pass.
        Is used for terminating parallel execution and threads in specific models.
        """
