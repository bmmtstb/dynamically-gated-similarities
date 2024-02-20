"""
Load, register, and initialize different optimizers.
"""

from typing import Type, Union

from torch import optim
from torch.optim import Optimizer

from dgs.utils.loader import get_instance, register_instance

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
    """Register a new optimizer to be used with custom configs.

    Args:
        optim_name: Name of the new optimizer, e.g. "CustomAdam".
            The name cannot be a value already present in ``OPTIMIZERS``.
        optimizer: The type / class of the optimizer to register.

    Raises:
        ValueError: If ``optim_name`` is in ``OPTIMIZERS.keys()`` or the ``optimizer`` is invalid.

    Examples::

        import torch
        from torch import optim
        class CustomAdam(optim.Optimizer):
            ...
        register_optimizer("CustomAdam", CustomAdam)
    """
    register_instance(name=optim_name, instance=optimizer, instances=OPTIMIZERS, inst_class=Optimizer)


def get_optimizer(instance: Union[str, Optimizer]) -> Type[Optimizer]:
    """Given the name or an instance of an optimizer, return the respective instance.

    Args:
        instance: Either the name of the optimizer, which has to be in ``OPTIMIZERS``,
            or a subclass of ``Optimizer``.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given optimizer.
    """
    return get_instance(instance=instance, instances=OPTIMIZERS, inst_class=Optimizer)
