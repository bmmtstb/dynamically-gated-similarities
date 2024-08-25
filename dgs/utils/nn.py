"""
Helper and utility methods for creating neural networks using PyTorch.
"""

from typing import Union

from torch import nn


def set_up_hidden_layer_sizes(
    input_size: int, output_size: int, hidden_sizes: Union[list[int], None] = None
) -> list[int]:
    """Given the input and output size of an FC-NN,
    create a list of the sizes containing each hidden layer in the network.
    There might be zero hidden layers.

    Params:
        input_size: The size of the input to the FC-Layers.
        output_size: Output-size of the FC-Layers.
        hidden_layers: The dimensionality of each hidden layer in this network. Default None means no hidden layers.

    Returns:
        The sizes of the hidden layers including input and output size.
    """
    layers: list[int] = [input_size]
    if not (hidden_sizes is None or len(hidden_sizes) == 0):
        for hidden_layer in hidden_sizes:
            layers.append(int(hidden_layer))
    layers.append(output_size)

    return layers


def fc_linear(hidden_layers: list[int], bias: Union[bool, list[bool]] = True) -> nn.Sequential:
    """Create a Network consisting of one or more fully connected linear layers
    with input and output sizes given by the ``hidden_layers``.

    Args:
        hidden_layers: A list containing the sizes of the input, hidden- and output layers.
            It is possible to use the :func:`set_up_hidden_layer_sizes` function to create this list.
        bias: Whether to use a bias in every layer.
            Can be a single value for the whole network or a list containing a value per layer (length - 1).
            Default is ``True``.

    Returns:
        A sequential model containing ``N-1`` fully-connected layers.
    """
    if isinstance(bias, bool):
        bias = [bias] * (len(hidden_layers) - 1)
    elif isinstance(bias, list):
        if len(bias) != (len(hidden_layers) - 1):
            raise ValueError(
                f"Length of bias {len(bias)} should be the same as the len(hidden_layers - 1) {len(hidden_layers) - 1}"
            )
    else:
        raise NotImplementedError(f"Bias should be a boolean or a list of booleans. Got: {bias}")

    if any(l <= 0 for l in hidden_layers):
        raise ValueError(f"Input, hidden or output size is <= 0. Got: {hidden_layers}")

    return nn.Sequential(
        *[
            nn.Linear(in_features=hidden_layers[i], out_features=hidden_layers[i + 1], bias=bias[i])
            for i in range(len(hidden_layers) - 1)
        ],
    )
