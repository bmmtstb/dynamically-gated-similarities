"""
Functions to load and manage torch loss functions.
"""

from typing import Type, Union

from torch import nn

from dgs.utils.types import Loss

try:
    # If torchreid is installed using `./dependencies/torchreid`
    # noinspection PyUnresolvedReferences LongLine
    from torchreid.losses import CrossEntropyLoss, TripletLoss
except ModuleNotFoundError:
    # if torchreid is installed using `pip install torchreid`
    # noinspection PyUnresolvedReferences
    from torchreid.reid.losses import CrossEntropyLoss, TripletLoss

LOSS_FUNCTIONS: dict[str, Type[Loss]] = {
    "L1Loss": nn.L1Loss,
    "NLLLoss": nn.NLLLoss,
    # "NLLLoss2d": nn.NLLLoss2d, deprecated
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "GaussianNLLLoss": nn.GaussianNLLLoss,
    "KLDivLoss": nn.KLDivLoss,
    "MSELoss": nn.MSELoss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "HuberLoss": nn.HuberLoss,
    "SoftMarginLoss": nn.SoftMarginLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MarginRankingLoss": nn.MarginRankingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
    "CTCLoss": nn.CTCLoss,
    "TorchreidTripletLoss": TripletLoss,
    "TorchreidCrossEntropyLoss": CrossEntropyLoss,
}


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
