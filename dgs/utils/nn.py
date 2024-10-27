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


def fc_linear(
    hidden_layers: list[int],
    bias: Union[bool, list[bool]] = True,
    act_func: Union[
        list[Union[str, None, nn.Module]],
        tuple[Union[str, None, nn.Module], ...],
    ] = None,
) -> nn.Sequential:
    """Create a Network consisting of one or more fully connected linear layers
    with input and output sizes given by the ``hidden_layers``.

    Args:
        hidden_layers: A list containing the sizes of the input, hidden- and output layers.
            It is possible to use the :func:`set_up_hidden_layer_sizes` function to create this list.
            The length of the hidden layers is denoted ``L``.
        bias: Whether to use a bias in every layer.
            Can be a single value for the whole network or a list of length ``L - 1`` containing one value per layer.
            Default is ``True``.
        act_func: A list containing the activation function after each of the fully connected layers.
            There can be a single activation function after every layer.
            Therefore, ``act_func`` should have a length of ``L``.
            Every value can either be the :class:`torch.nn.Module` or the string representing the activation function.
            E.g. "ReLU" for :class:`~torch.nn.ReLU`
            Defaults to adding no activation functions.

    Returns:
        A sequential model containing ``N-1`` fully-connected layers.
    """
    # pylint: disable=too-many-branches

    L = len(hidden_layers)
    # validate bias
    if isinstance(bias, bool):
        bias = [bias] * (L - 1)
    elif isinstance(bias, list):
        if len(bias) != (L - 1):
            raise ValueError(f"Length of bias {len(bias)} should be the same as L - 1 but got: {L - 1}")
    else:
        raise NotImplementedError(f"Bias should be a boolean or a list of booleans. Got: {bias}")

    # validate activation functions
    if act_func is None:
        act_func = [None] * (L - 1)
    elif isinstance(act_func, str):
        act_func = [act_func] * (L - 1)

    if not isinstance(act_func, (list, tuple)) or len(act_func) != (L - 1):
        raise ValueError(f"The activation functions should be a list of length L - 1, but got: {act_func}")
    if not all(
        af is None or isinstance(af, str) or (isinstance(af, type) and issubclass(af, nn.Module)) for af in act_func
    ):
        raise ValueError(f"Expected all activation functions to be None, strings, or a nn.Module, but got: {act_func}")

    # validate hidden layers
    if any(l <= 0 for l in hidden_layers):
        raise ValueError(f"Input, hidden or output size is <= 0. Got: {hidden_layers}")

    layers = []

    for i in range(L - 1):
        layers.append(nn.Linear(in_features=hidden_layers[i], out_features=hidden_layers[i + 1], bias=bias[i]))

        a_i = act_func[i]
        if a_i is not None:
            if isinstance(a_i, str):
                try:
                    a_i = getattr(nn, a_i)
                except AttributeError as e:
                    raise AttributeError(f"Tried to load non-existent activation function '{a_i}'.") from e
            layers.append(a_i())  # make sure to call / instantiate the function here

    return nn.Sequential(*layers)
