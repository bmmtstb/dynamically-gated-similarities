"""
Utility functions for loading instances.

This module does not provide functionality for loading Modules.
"""

from typing import Type, TypeVar

from dgs.utils.constants import MODULE_TYPES
from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.types import Instance

I = TypeVar("I")


def register_instance(
    name: str, instance: Type[I], instances: dict[str, Type[I]], inst_class: type, call: bool = True
) -> None:
    """Given an instance with a name, add it to the available instances.

    Args:
        name: The name of the instance.
            Cannot be a value already present in ``instances``.
        instance: The instance that should be added to ``instances``.
        instances: A dictionary containing a mapping from instance names to instance classes.
        inst_class: The class the instance should have.
        call: Whether the instance should be callable.
            Default True.

    Raises:
        ValueError if either the name exists in instances or the instance has incorrect properties.
    """
    if name in instances:
        raise KeyError(
            f"The given name '{name}' already exists within the registered instances. "
            f"Please choose another name excluding '{list(instances.keys())}'."
        )
    if call and not callable(instance):
        raise TypeError("The given instance is not callable.")
    if not (isinstance(instance, type) and issubclass(instance, inst_class)):
        raise TypeError(f"The given instance is not a valid subclass of type '{inst_class}'. Got: {instance}")
    instances[name] = instance


def get_instance_from_name(name: str, instances: dict[str, Type[I]]) -> Type[I]:
    """Given the name of an instance and the dict containing a mapping from name to class, get the class.

    Args:
        name: The name of the instance to add to ``instances``.
        instances: A dictionary containing a mapping from instance name to instance class.

    Returns:
        The class-type of the instance.

    Raises:
        ValueError if the instance name is not present in ``instances``.
    """
    if name not in instances:
        raise KeyError(f"Instance '{name}' is not defined in '{list(instances.keys())}'.")
    return instances[name]


def get_instance(instance: Instance, instances: dict[str, Type[I]], inst_class: type) -> Type[I]:
    """

    Args:
        instance: Either the name of the instance, which has to be in ``instances``,
            or a subclass of `Optimizer`.
        instances: A dictionary containing a mapping from instance names to instance classes.
        inst_class: The class the instance should have.


    Raises:
        ValueError: If the instance has the wrong type.
        InvalidParameterException: If the instance is neither string nor of type ``inst_class``.

    Returns:
        The class-type of the given instance.
    """
    if isinstance(instance, str):
        return get_instance_from_name(name=str(instance), instances=instances)
    if isinstance(instance, type) and issubclass(instance, inst_class):
        return instance
    raise InvalidParameterException(f"Instance {instance} is neither string nor a subclass of '{inst_class}'")


def get_registered_classes(module_type: str) -> dict[str, type]:
    """

    Args:
        module_type: The type of module to get all the registered names from.

    Returns:
        A set containing all registered names.
    """
    # pylint: disable=too-many-branches,import-outside-toplevel,cyclic-import

    if module_type not in MODULE_TYPES:
        raise ValueError(f"The instance class name '{module_type}' could not be found.")

    if module_type == "alpha":
        from dgs.models.alpha import ALPHA_MODULES as modules
    elif module_type == "combine":
        from dgs.models.combine import COMBINE_MODULES as modules
    elif module_type == "dataset":
        from dgs.models.dataset import DATASETS as modules
    elif module_type == "dataloader":
        raise ValueError("dataloaders can not be registered. Did you mean dataset?")
    elif module_type == "dgs":
        from dgs.models.dgs import DGS_MODULES as modules
    elif module_type == "embedding_generator":
        from dgs.models.embedding_generator import EMBEDDING_GENERATORS as modules
    elif module_type == "engine":
        from dgs.models.engine import ENGINES as modules
    elif module_type == "loss":
        from dgs.models.loss import LOSS_FUNCTIONS as modules
    elif module_type == "metric":
        from dgs.models.metric import METRICS as modules
    elif module_type == "optimizer":
        from dgs.models.optimizer import OPTIMIZERS as modules
    elif module_type == "similarity":
        from dgs.models.similarity import SIMILARITIES as modules
    elif module_type == "submission":
        from dgs.models.submission import SUBMISSION_FORMATS as modules
    else:
        raise NotImplementedError

    return modules


def get_registered_class_names(module_type: str) -> set[str]:
    """Get the names of all classes registered in a given module.

    Args:
        module_type: The type of module to get all the registered names from.

    Returns:
        A set containing all registered names.
    """
    return set(get_registered_classes(module_type=module_type).keys())


def get_registered_class_types(module_type: str) -> set[type]:
    """Get the class types of all classes registered in a given module.

    Args:
        module_type: The type of module to get all the registered types from.

    Returns:
        A set containing all registered types.
    """
    return set(get_registered_classes(module_type=module_type).values())
