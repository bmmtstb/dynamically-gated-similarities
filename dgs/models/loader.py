"""
Load and register modules.
"""

from typing import Union

from torch.utils.data import DataLoader as TDataLoader, Dataset as TDataset

from dgs.models.dataset.dataset import BaseDataset, dataloader_validations
from dgs.utils.config import DEF_VAL, get_sub_config
from dgs.utils.constants import MODULE_TYPES
from dgs.utils.exceptions import InvalidConfigException
from dgs.utils.state import collate_list_of_history, collate_lists, collate_states
from dgs.utils.types import Config, NodePath

__all__ = ["module_loader", "register_module", "get_data_loader"]


def module_loader(config: Config, module_type: str, key: Union[NodePath, str], *_, **kwargs) -> any:
    """Load a given module and pass down the configuration

    Args:
        config: The configuration of the current tracker
        module_type: The type of the module.
        key: Name of the key in the configuration that contains the parameters for this module.
            Can be a list of names for nested configurations.
        kwargs: Additional keyword arguments passed to the module during initialization.

    Returns:
        Initialized instance of the submodule with its config.
    """
    # This model will have one branch for every module
    # pylint: disable=too-many-branches,import-outside-toplevel,cyclic-import

    if module_type not in MODULE_TYPES:
        raise ValueError(f"The module type: '{module_type}' could not be found.")

    path: NodePath = [key] if isinstance(key, str) else key
    sub_cfg: Config = get_sub_config(config, path)
    if "module_name" not in sub_cfg:
        raise InvalidConfigException(f"Module at path '{path}' does not contain a module name.")
    module_name: str = sub_cfg["module_name"]

    # Module import and initialization
    if module_type == "alpha":
        from dgs.models.alpha import get_alpha_module

        m = get_alpha_module(module_name)
    elif module_type == "combine":
        from dgs.models.combine import get_combine_module

        m = get_combine_module(module_name)
    elif module_type == "dataloader":
        return get_data_loader(config=config, path=path)
    elif module_type == "dataset":
        # special case: a generally concatenated dataset
        if module_name.startswith("Concat_"):
            from dgs.models.dataset import get_multi_dataset

            return get_multi_dataset(config=config, path=path, ds_name=module_name[7:], concat=True)
        if module_name.startswith("List_"):
            from dgs.models.dataset import get_multi_dataset

            return get_multi_dataset(config=config, path=path, ds_name=module_name[5:], concat=False)
        from dgs.models.dataset import get_dataset

        m = get_dataset(module_name)
    elif module_type == "dgs":
        from dgs.models.dgs import get_dgs_module

        m = get_dgs_module(module_name)
    elif module_type == "embedding_generator":
        from dgs.models.embedding_generator import get_embedding_generator

        m = get_embedding_generator(module_name)
    elif module_type == "engine":
        from dgs.models.engine import get_engine

        m = get_engine(module_name)
    elif module_type == "similarity":
        from dgs.models.similarity import get_similarity_module

        m = get_similarity_module(module_name)
    elif module_type == "submission":
        from dgs.models.submission import get_submission

        m = get_submission(module_name)
    else:
        raise NotImplementedError(f"Something went wrong while loading the module '{module_type}'")

    # instantiate module with its configuration and path
    return m(config=config, path=path, **kwargs)


def register_module(name, new_module, module_type: str) -> None:
    r"""Register a new module.

    Args:
        name: The name under which to register the new module.
        new_module: The type of the new module to register.
        module_type: The type of module instance to register. Has to be in :data:`MODULE_TYPES`.

    Raises:
        ValueError: If the instance class name is invalid.

    Examples::

        from torch import nn
        from dgs.models import register_module
        class CustomNNLLoss(Loss):
            def __init__(...):
                ...
            def forward(self, input: torch.Tensor, target: torch.Tensor):
                return ...
        register_module(name="CustomNNLLoss", new_module=CustomNNLLoss, inst_class_name="loss")
    """
    # pylint: disable=too-many-branches,import-outside-toplevel

    if module_type not in MODULE_TYPES:
        raise ValueError(f"The instance class name '{module_type}' could not be found.")

    if module_type == "combine":
        from dgs.models.combine import register_combine_module

        register_combine_module(name=name, new_combine=new_module)
    elif module_type == "dataloader":
        raise ValueError("You can not register a new dataloader. Did you want to register a dataset?")
    elif module_type == "dataset":
        from dgs.models.dataset import register_dataset

        register_dataset(name=name, new_ds=new_module)
    elif module_type == "dgs":
        from dgs.models.dgs import register_dgs_module

        register_dgs_module(name=name, new_dgs=new_module)
    elif module_type == "embedding_generator":
        from dgs.models.embedding_generator import register_embedding_generator

        register_embedding_generator(name=name, new_eg=new_module)
    elif module_type == "engine":
        from dgs.models.engine import register_engine

        register_engine(name=name, new_engine=new_module)
    elif module_type == "loss":
        from dgs.models.loss import register_loss_function

        register_loss_function(name=name, new_loss=new_module)
    elif module_type == "metric":
        from dgs.models.metric import register_metric

        register_metric(name=name, new_metric=new_module)
    elif module_type == "optimizer":
        from dgs.models.optimizer import register_optimizer

        register_optimizer(name=name, new_optimizer=new_module)

    elif module_type == "similarity":
        from dgs.models.similarity import register_similarity_module

        register_similarity_module(name=name, new_similarity=new_module)
    elif module_type == "submission":
        from dgs.models.submission import register_submission

        register_submission(name=name, new_sub=new_module)
    else:
        raise NotImplementedError


def get_data_loader(config: Config, path: NodePath) -> TDataLoader:
    """Set up a torch data loader with some params from config.

    Args:
        config: The overall configuration of the algorithm.
        path: The node path to the params of this DataLoader.

    Returns:
        A `~.DataLoader` object for the given dataset.
    """
    ds: BaseDataset = module_loader(config=config, module_type="dataset", key=path)

    # validate data loader params on regular BaseDataset, a concatenated dataset should have been validated elsewhere
    if isinstance(ds, BaseDataset):
        ds.validate_params(dataloader_validations)
        params = ds.params
    else:
        assert isinstance(ds, TDataset)
        params = get_sub_config(config=config, path=path)

    batch_size: int = params.get("batch_size", DEF_VAL["dataloader"]["batch_size"])
    drop_last: bool = params.get("drop_last", DEF_VAL["dataloader"]["drop_last"])
    shuffle: bool = params.get("shuffle", DEF_VAL["dataloader"]["shuffle"])

    if "collate_fn" not in params:
        collate_fn = collate_states
    elif params["collate_fn"] == "lists" or params["collate_fn"] is None:
        collate_fn = collate_lists
    elif params["collate_fn"] == "states":
        collate_fn = collate_states
    elif params["collate_fn"] == "history":
        collate_fn = collate_list_of_history
    else:
        raise NotImplementedError(f"Collate function '{params['collate_fn']}' not implemented.")

    return TDataLoader(
        dataset=ds,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=params.get("workers", DEF_VAL["dataloader"]["workers"]),
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
