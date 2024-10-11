"""
An alpha module combining other alpha modules.
"""

import torch as t

from dgs.models.alpha.alpha import BaseAlphaModule
from dgs.models.loader import module_loader
from dgs.utils.config import get_sub_config, insert_into_config
from dgs.utils.state import get_ds_data_getter, State
from dgs.utils.types import Config, DataGetter, NodePath, Validations

sequential_combined_validations: Validations = {
    "paths": ["NodePaths", ("longer eq", 1)],
    # optional
}


class SequentialCombinedAlpha(BaseAlphaModule):
    """An alpha module sequentially combining multiple other :class:`BaseAlphaModule` s.
    The first module will load the data from the :class:`State`.
    In every other module the resulting :class:`.Tensor` will be inserted into the forward call.

    Params
    ------

    paths (list[NodePath]):
        A :class:`NodePath` pointing to the configuration of each of the other :class:`~BaseAlphaModule` s.
        Only the first module needs to have its `name` property set,
        all other layers will use the result returned by the previous layer.

    Optional Params
    ---------------

    """

    model: t.nn.Sequential

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config=config, path=path)

        self.validate_params(sequential_combined_validations)

        # get the name parameter from the first layer
        first_cfg = get_sub_config(config=config, path=self.params["paths"][0])
        if "name" not in first_cfg or first_cfg["name"] in [None, ""]:
            raise ValueError(f"Configuration of first module must have `name` key, but got: {first_cfg}")
        self.data_getter: DataGetter = get_ds_data_getter(first_cfg["name"])

        # get all modules
        modules: list[BaseAlphaModule] = []
        for sub_path in self.params["paths"]:
            # set name of all the submodules to empty string
            # can be done for the first module too, because the data_getter is already set
            new_cfg = insert_into_config(path=sub_path, value={"name": ""}, original=config, copy=True)
            modules.append(module_loader(config=new_cfg, module_type="alpha", key=sub_path))

        self.register_module(name="model", module=self.configure_torch_module(t.nn.Sequential(*modules)))

    def forward(self, s: State) -> t.Tensor:
        inpt = self.get_data(s)
        for sub_models in self.model:
            inpt = sub_models.sub_forward(inpt)
        return inpt

    def get_data(self, s: State) -> tuple[any, ...]:
        return self.data_getter(s)
