"""
Functions to load and manage torch loss functions.
"""

from typing import Type, Union

import torch
from torch import nn

from dgs.utils.types import Loss

try:
    # If torchreid is installed using `./dependencies/torchreid`
    # noinspection PyUnresolvedReferences LongLine
    from torchreid.losses import CrossEntropyLoss as TorchreidCEL, TripletLoss as TorchreidTL
except ModuleNotFoundError:
    # if torchreid is installed using `pip install torchreid`
    # noinspection PyUnresolvedReferences
    from torchreid.reid.losses import CrossEntropyLoss as TorchreidCEL, TripletLoss as TorchreidTL


def register_loss_function(loss_name: str, loss_function: Type[Loss]) -> None:
    """Register a new loss function.

    Args:
        loss_name: Name of the new loss function, e.g. "CustomNNLLoss".
            Cannot be a value that is already in `LOSS_FUNCTIONS`.
        loss_function: The type of loss function to register.

    Raises:
        ValueError: If `loss_name` is in `LOSS_FUNCTIONS.keys()` or the `loss_function` is invalid.

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
    if loss_name in LOSS_FUNCTIONS:
        raise ValueError(
            f"The given name '{loss_name}' already exists, "
            f"please choose another name excluding {LOSS_FUNCTIONS.keys()}."
        )
    if not (callable(loss_function) and isinstance(loss_function, type) and issubclass(loss_function, Loss)):
        raise ValueError(f"The given loss function is no callable or no subclass of Loss. Got: {loss_function}")
    LOSS_FUNCTIONS[loss_name] = loss_function


def get_loss_from_name(name: str) -> Type[Loss]:
    """Given a name of a loss function, that is in `LOSS_FUNCTIONS`, return an instance of it.

    Params:
        name: The name of the loss function.

    Raises:
        ValueError: If the loss function does not exist in `LOSS_FUNCTIONS`.

    Returns:
        An instance of the loss function.
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{name}' is not defined.")
    return LOSS_FUNCTIONS[name]


def get_loss_function(instance: Union[str, callable]) -> Type[Loss]:
    """

    Args:
        instance: Either the name of the loss function, which has to be in `LOSS_FUNCTIONS`,
            or a subclass of `Loss`.

    Raises:
        ValueError: If the instance has the wrong type.

    Returns:
        The class of the given loss function.
    """
    if isinstance(instance, str):
        return get_loss_from_name(instance)
    if isinstance(instance, type) and issubclass(instance, Loss):
        return instance
    raise ValueError(f"Loss function '{instance}' is neither string nor subclass of 'Loss'.")


class CrossEntropyLoss(Loss):
    """Compute the Cross Entropy Loss after computing the LogSoftmax on the input data."""

    def __init__(self, **kwargs):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, inputs: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        """Given predictions of shape ``[B x nof_classes]`` and targets of shape ``[B]``
        compute and return the CrossEntropyLoss.
        """
        logits = self.log_softmax(inputs)
        return self.cross_entropy_loss(logits, targets)


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
