"""
A class for alpha modules with one or multiple fully connected layers.
"""

import torch as t

from dgs.models.alpha.alpha import BaseAlphaModule
from dgs.utils.config import DEF_VAL
from dgs.utils.nn import fc_linear
from dgs.utils.state import get_ds_data_getter, State
from dgs.utils.types import Config, DataGetter, NodePath, Validations

fc_validations: Validations = {
    "name": [str],
    "hidden_layers": [list, ("forall", int)],
    "bias": [("any", [bool, ("all", [list, ("forall", bool)])])],
    # optional
    "act_func": ["optional", ("any", [("isinstance", str), "None", ("isinstance", t.nn.Module)])],
}


class FullyConnectedAlpha(BaseAlphaModule):
    """An alpha module consisting of ``L - 1`` fully connected layers.
    Each layer can have a custom bias and activation function.

    Params
    ------

    name (str):
        The name of the attribute or getter function used to retrieve the input data from the state.
    hidden_layers (list[int]):
        The sizes of each of the hidden layers, including the size of the data.
        Has length ``L``.
    bias (Union[bool, list[bool]):
        Whether each of the respective layers should have values for the bias.
        Has length ``L - 1``.

    Optional Params
    ---------------

    act_func (list[Union[str, None, nn.Module]]):
        A list containing the activation functions placed after each of the layers.
        Has length ``L - 1``.
        Default ``DEF_VAL.alpha.act_func``.

    """

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config=config, path=path)

        self.validate_params(fc_validations)

        self.data_getter: DataGetter = get_ds_data_getter(self.params["name"])

        model = fc_linear(
            hidden_layers=self.params["hidden_layers"],
            bias=self.params["bias"],
            act_func=self.params.get("act_func", DEF_VAL["alpha"]["act_func"]),
        )
        self.register_module(name="model", module=self.configure_torch_module(model))

    def forward(self, s: State) -> t.Tensor:
        return self.model(self.get_data(s))

    def get_data(self, s: State) -> t.Tensor:
        return self.data_getter(s)
