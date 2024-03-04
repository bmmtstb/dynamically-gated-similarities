"""
Functions to load and manage torch and custom loss functions.
"""

import warnings
from typing import Type

from torch import nn

from dgs.utils.loader import get_instance, register_instance
from dgs.utils.types import Instance, Loss
from .loss import CrossEntropyLoss

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Cython evaluation.*is unavailable", category=UserWarning)
    try:
        # If torchreid is installed using `./dependencies/torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.losses import CrossEntropyLoss as TorchreidCEL, TripletLoss as TorchreidTL
    except ModuleNotFoundError:
        # if torchreid is installed using `pip install torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.reid.losses import CrossEntropyLoss as TorchreidCEL, TripletLoss as TorchreidTL


LOSS_FUNCTIONS: dict[str, Type[Loss]] = {
    # own
    "CrossEntropyLoss": CrossEntropyLoss,
    # pytorch
    "TorchL1Loss": nn.L1Loss,
    "TorchNLLLoss": nn.NLLLoss,
    "TorchPoissonNLLLoss": nn.PoissonNLLLoss,
    "TorchGaussianNLLLoss": nn.GaussianNLLLoss,
    "TorchKLDivLoss": nn.KLDivLoss,
    "TorchMSELoss": nn.MSELoss,
    "TorchBCELoss": nn.BCELoss,
    "TorchBCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "TorchHingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "TorchMultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "TorchSmoothL1Loss": nn.SmoothL1Loss,
    "TorchHuberLoss": nn.HuberLoss,
    "TorchSoftMarginLoss": nn.SoftMarginLoss,
    "TorchCrossEntropyLoss": nn.CrossEntropyLoss,
    "TorchMultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "TorchCosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "TorchMarginRankingLoss": nn.MarginRankingLoss,
    "TorchMultiMarginLoss": nn.MultiMarginLoss,
    "TorchTripletMarginLoss": nn.TripletMarginLoss,
    "TorchTripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    "TorchCTCLoss": nn.CTCLoss,
    # TorchReid
    "TorchreidTripletLoss": TorchreidTL,
    "TorchreidCrossEntropyLoss": TorchreidCEL,
}


def register_loss_function(name: str, new_loss: Type[Loss]) -> None:
    """Register a new loss function to be used with custom configs.

    Args:
        name: Name of the new loss function, e.g. "CustomNNLLoss".
            The name cannot be a value that is already in :data:``LOSS_FUNCTIONS``.
        new_loss: The type of loss function to register.

    Raises:
        ValueError: If ``loss_name`` is in :data:``LOSS_FUNCTIONS.keys()`` or the instance is invalid.

    Examples::

        import torch
        from torch import nn
        class CustomNNLLoss(Loss):
            def __init__(...):
                ...
            def forward(self, input: torch.Tensor, target: torch.Tensor):
                return ...
        register_loss_function("CustomNNLLoss", CustomNNLLoss)
    """
    register_instance(name=name, instance=new_loss, instances=LOSS_FUNCTIONS, inst_class=Loss)


def get_loss_function(instance: Instance) -> Type[Loss]:
    """Given the name or an instance of a loss function, return the respective instance.

    Args:
        instance: Either the name of the loss function, which has to be in :data:``LOSS_FUNCTIONS``,
            or a subclass of :class:``~.Loss``.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given loss function.
    """
    return get_instance(instance=instance, instances=LOSS_FUNCTIONS, inst_class=Loss)
