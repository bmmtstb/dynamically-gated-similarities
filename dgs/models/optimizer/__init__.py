"""
Load, register, and initialize different optimizers.
"""

from typing import Type

from torch import optim
from torch.optim import Optimizer

from dgs.utils.loader import get_instance, register_instance
from dgs.utils.types import Instance

__all__ = ["OPTIMIZERS", "register_optimizer", "get_optimizer"]

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


def register_optimizer(name: str, new_optimizer: Type[Optimizer]) -> None:
    """Register a new optimizer to be used with custom configs.

    Args:
        name: Name of the new optimizer, e.g. "CustomAdam".
            The name cannot be a value already present in :data:``OPTIMIZERS``.
        new_optimizer: The type / class of the optimizer to register.

    Raises:
        ValueError: If ``optim_name`` is in :data:``OPTIMIZERS.keys()`` or the instance is invalid.

    Examples::

        from torch import optim
        class CustomAdam(optim.Optimizer):
            ...
        register_optimizer("CustomAdam", CustomAdam)
    """
    register_instance(name=name, instance=new_optimizer, instances=OPTIMIZERS, inst_class=Optimizer)


def get_optimizer(instance: Instance) -> Type[Optimizer]:
    """Given the name or an instance of an optimizer, return the respective instance.

    Args:
        instance: Either the name of the optimizer, which has to be in :data:``OPTIMIZERS``,
            or a subclass of :class:`.Optimizer`.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given optimizer.
    """
    return get_instance(instance=instance, instances=OPTIMIZERS, inst_class=Optimizer)
