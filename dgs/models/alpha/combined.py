"""
An alpha module combining other alpha modules.
"""

import torch as t

from dgs.models.alpha.alpha import BaseAlphaModule
from dgs.models.loader import module_loader
from dgs.utils.config import insert_into_config
from dgs.utils.state import get_ds_data_getter, State
from dgs.utils.types import Config, DataGetter, NodePath, Validations

sequential_combined_validations: Validations = {
    "paths": [list, ("longer eq", 1), ("forall", ("any", [str, dict, ("all", [list, ("forall", str)])]))],
    "name": [str],
    # optional
}


class SequentialCombinedAlpha(BaseAlphaModule):
    """An alpha module sequentially combining multiple other :class:`BaseAlphaModule` s.
    First load the data from the :class:`State` using `name`.
    Then insert the resulting :class:`.Tensor` into the forward call of the respective next model.

    Params
    ------

    paths (list[str, NodePath]):
        A list containing either :class:`NodePath` s pointing to the configuration of a :class:`~BaseAlphaModule`
        or the name of a function from `torch.nn` (e.g. 'Flatten', 'ReLU', ...).
        All submodules do not need to have the "name" property,
        because all other layers will use the result returned by the previous layer.
    name (str):
        The name of the attribute or getter function used to retrieve the input data from the state.

    Optional Params
    ---------------

    """

    model: t.nn.Sequential

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config=config, path=path)

        self.validate_params(sequential_combined_validations)

        self.data_getter: DataGetter = get_ds_data_getter(self.params["name"])

        # get all modules
        modules: list[BaseAlphaModule] = []
        for sub_path in self.params["paths"]:
            if isinstance(sub_path, list):
                # set name of all the submodules to empty string
                # can be done for the first module too, because the data_getter is already set
                new_cfg = insert_into_config(path=sub_path, value={"name": ""}, original=config, copy=True)
                modules.append(module_loader(config=new_cfg, module_type="alpha", key=sub_path))
            elif isinstance(sub_path, str):
                try:
                    modules.append(getattr(t.nn, sub_path)())
                except AttributeError as e:
                    raise AttributeError(f"Tried to load non-existent torch module '{sub_path}'.") from e
            elif isinstance(sub_path, dict):
                if len(sub_path) > 1:
                    raise ValueError(f"Expected submodule config to be a single dict, got: {sub_path}")
                k, v = list(sub_path.keys())[0], list(sub_path.values())[0]
                if not isinstance(v, dict):
                    raise NotImplementedError(f"Expected submodule parameters to be a dict, got: {v}")
                try:
                    modules.append(getattr(t.nn, k)(**v))
                except AttributeError as e:
                    raise AttributeError(f"Tried to load non-existent torch module '{sub_path}'.") from e
            else:
                raise NotImplementedError(f"Expected list or str, got: {sub_path}")

        self.register_module(name="model", module=self.configure_torch_module(t.nn.Sequential(*modules)))

    def forward(self, s: State) -> t.Tensor:
        """Forward call for sequential model calls the next layer with the output of the previous layer.
        Works for :class:`BaseAlphaModule` s and any arbitrary model from `torch.nn`.
        """
        inpt = self.get_data(s)
        for sub_models in self.model:
            inpt = sub_models(inpt)
        return inpt

    def get_data(self, s: State) -> tuple[any, ...]:
        return self.data_getter(s)
