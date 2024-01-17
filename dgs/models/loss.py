from torch import nn


LOSS_FUNCTIONS: dict[str, callable] = {
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
}


def get_loss_from_name(name: str) -> nn.Module:
    """Given a name of a loss function, that is in ..., return an instance of it.

    Params:
        name: The name of the loss function.

    Returns:
        An instance of the loss function.
    """
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{name}' is not defined.")
    return LOSS_FUNCTIONS[name]
