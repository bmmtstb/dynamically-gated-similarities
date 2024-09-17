"""
Load, register, and initialize different learning rate schedulers.
"""

from typing import Type

from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

from dgs.utils.loader import get_instance, register_instance
from dgs.utils.types import Instance, Scheduler

__all__ = ["SCHEDULERS", "register_scheduler", "get_scheduler"]

SCHEDULERS: dict[str, Type[Scheduler]] = {
    "LambdaLR": LambdaLR,
    "MultiplicativeLR": MultiplicativeLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ConstantLR": ConstantLR,
    "LinearLR": LinearLR,
    "ExponentialLR": ExponentialLR,
    "PolynomialLR": PolynomialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ChainedScheduler": ChainedScheduler,
    "SequentialLR": SequentialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CyclicLR": CyclicLR,
    "OneCycleLR": OneCycleLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
}


def register_scheduler(sched_name: str, scheduler: Type[Scheduler]) -> None:
    """Register a new learning-rate scheduler to be used with custom configs.

    Args:
        sched_name: Name of the new scheduler, e.g. "StepwiseIncrement".
            The name cannot be a value already present in ``SCHEDULERS``.
        scheduler: The type / class of the learning rate scheduler to register.

    Raises:
        ValueError: If ``sched_name`` is in ``SCHEDULERS.keys()`` or the ``scheduler`` is invalid.

    Examples::

        from dgs.utils.types import Scheduler
        class CustomLinear(Scheduler):
            ...
        register_scheduler("CustomLinear", CustomLinear)
    """

    register_instance(name=sched_name, instance=scheduler, instances=SCHEDULERS, inst_class=Scheduler)


def get_scheduler(instance: Instance) -> Type[Scheduler]:
    """Given the name or an instance of a learning-rate scheduler, return the respective instance.

    Args:
        instance: Either the name of the scheduler, which has to be in ``SCHEDULERS``,
            or a subclass of ``Scheduler``.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given scheduler.
    """
    return get_instance(instance=instance, instances=SCHEDULERS, inst_class=Scheduler)
