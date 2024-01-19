"""
Functions to load and manage torch optimizers.
"""
from typing import Type, Union

from torch import optim
from torch.optim import Optimizer

OPTIMIZERS: dict[str, Type[Optimizer]] = {
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SparseAdam": optim.SparseAdam,
    "Adamax": optim.Adamax,
    "ASGD": optim.ASGD,
    # "LBFGS": optim.LBFGS,  # I don't want to handle closures...
    "NAdam": optim.NAdam,
    "RAdam": optim.RAdam,
    "RMSprop": optim.RMSprop,
    "Rprop": optim.Rprop,
    "SGD": optim.SGD,
}


def register_optimizer(optim_name: str, optimizer: Type[Optimizer]) -> None:
    """Register a new optimizer.

    Args:
        optim_name: Name of the new optimizer, e.g. "CustomAdam".
            Cannot be a value that is already in `OPTIMIZERS`.
        optimizer: The type / class of the optimizer to register.

    Raises:
        ValueError: If `optim_name` is in `OPTIMIZERS.keys()` or the `optimizer` is invalid.

    Examples::

        import torch
        from torch import optim
        class CustomAdam(optim.Optimizer):
            ...
        register_optimizer("CustomAdam", CustomAdam)
    """
    if optim_name in OPTIMIZERS:
        raise ValueError(
            f"The given name '{optim_name}' already exists, "
            f"please choose another name excluding {OPTIMIZERS.keys()}."
        )
    if not (callable(optimizer) and isinstance(optimizer, type) and issubclass(optimizer, Optimizer)):
        raise ValueError(f"The given optimizer is no callable or no subclass of Optimizer. Got: {optimizer}")
    OPTIMIZERS[optim_name] = optimizer


def get_optim_from_name(name: str) -> Type[Optimizer]:
    """Given the name of an optimizer, that is in `OPTIMIZERS`, return an instance of it.

    Params:
        name: The name of the optimizer.

    Raises:
        ValueError: If the optimizer does not exist in `OPTIMIZERS`.

    Returns:
        The type of the optimizer.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' is not defined.")
    return OPTIMIZERS[name]


def get_optimizer(instance: Union[str, Optimizer]) -> Type[Optimizer]:
    """

    Args:
        instance: Either the name of the optimizer, which has to be in `OPTIMIZERS`,
            or a subclass of `Optimizer`.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given optimizer.
    """
    if isinstance(instance, str):
        return get_optim_from_name(instance)
    if isinstance(instance, type) and issubclass(instance, Optimizer):
        return instance
    raise ValueError(f"Optimizer '{instance}' is neither string nor subclass of 'Optimizer'.")
