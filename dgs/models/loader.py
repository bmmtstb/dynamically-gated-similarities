"""
Load and register modules.
"""

from typing import Type, TypeVar, Union

import torch.nn
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from dgs.models.dataset.dataset import BaseDataset, dataloader_validations
from dgs.models.module import BaseModule
from dgs.utils.config import DEF_VAL, get_sub_config
from dgs.utils.exceptions import InvalidConfigException
from dgs.utils.state import collate_lists, collate_states
from dgs.utils.types import Config, NodePath

M = TypeVar("M", bound=BaseModule)


def module_loader(
    config: Config, module_class: str, key: Union[NodePath, str], *_, **kwargs
) -> Union[M, TorchDataLoader]:
    """Load a given module and pass down the configuration

    Args:
        config: The configuration of the current tracker
        module_class: The type of the module.
        key: Name of the key in the configuration that contains the parameters for this module.
            Can be a list of names for nested configurations.
        kwargs: Additional keyword arguments passed to the module during initialization.

    Returns:
        Initialized instance of the submodule with its config.
    """
    # This model will have one branch for every module
    # pylint: disable=too-many-branches,import-outside-toplevel
    path: NodePath = [key] if isinstance(key, str) else key
    sub_cfg: Config = get_sub_config(config, path)
    if "module_name" not in sub_cfg:
        raise InvalidConfigException(f"Module at path '{path}' does not contain a module name.")
    module_name: str = sub_cfg["module_name"]

    m: Type[M]

    # Module import and initialization
    if module_class == "combine":
        from dgs.models.combine import get_combine_module

        m = get_combine_module(module_name)
    elif module_class == "dataloader":
        return get_data_loader(config=config, path=path)
    elif module_class == "dataset":
        # special case: the concatenated PT21 dataset is loaded via function not class
        if module_name.startswith("PoseTrack21"):
            from dgs.models.dataset.posetrack21 import get_pose_track_21

            return get_pose_track_21(
                config=config, path=path, ds_name="image" if module_name.endswith("Image") else "bbox"
            )
        from dgs.models.dataset import get_dataset

        m = get_dataset(module_name)
    elif module_class == "dgs":
        from dgs.models.dgs import get_dgs_module

        m = get_dgs_module(module_name)
    elif module_class == "embedding_generator":
        from dgs.models.embedding_generator import get_embedding_generator

        m = get_embedding_generator(module_name)
    elif module_class == "engine":
        from dgs.models.engine import get_engine

        m = get_engine(module_name)
        path = []
    elif module_class == "similarity":
        from dgs.models.similarity import get_similarity_module

        m = get_similarity_module(module_name)
    else:
        raise NotImplementedError(f"Something went wrong while loading the module '{module_class}'")

    # instantiate module with its configuration and path
    return m(config=config, path=path, **kwargs)


def register_module(name, new_module: Union[Type[M], Type[torch.nn.Module]], inst_class_name: str) -> None:
    r"""Register a new module.

    Args:
        name: The name under which to register the new module.
        new_module: The type of the new module to register.
        inst_class_name: The class-name of the module instance to register.

    Raises:
        ValueError: If the instance class name is invalid.

    Examples::

        import torch
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

    if inst_class_name == "combine":
        from dgs.models.combine import register_combine_module

        register_combine_module(name=name, new_combine=new_module)
    elif inst_class_name == "dataset":
        from dgs.models.dataset import register_dataset

        register_dataset(name=name, new_ds=new_module)
    elif inst_class_name == "dgs":
        from dgs.models.dgs import register_dgs_module

        register_dgs_module(name=name, new_dgs=new_module)
    elif inst_class_name == "embedding_generator":
        from dgs.models.embedding_generator import register_embedding_generator

        register_embedding_generator(name=name, new_eg=new_module)
    elif inst_class_name == "engine":
        from dgs.models.engine import register_engine

        register_engine(name=name, new_engine=new_module)
    elif inst_class_name == "loss":
        from dgs.models.loss import register_loss_function

        register_loss_function(name=name, new_loss=new_module)
    elif inst_class_name == "metric":
        from dgs.models.metric import register_metric

        register_metric(name=name, new_metric=new_module)
    elif inst_class_name == "optimizer":
        from dgs.models.optimizer import register_optimizer

        register_optimizer(name=name, new_optimizer=new_module)

    elif inst_class_name == "similarity":
        from dgs.models.similarity import register_similarity_module

        register_similarity_module(name=name, new_similarity=new_module)
    else:
        raise ValueError(f"The instance class name '{inst_class_name}' could not be found.")


def get_data_loader(config: Config, path: NodePath) -> TorchDataLoader:
    """Set up a torch data loader with some params from config.

    Args:
        config: The overall configuration of the algorithm.
        path: The node path to the params of this DataLoader.

    Returns:
        A `~.DataLoader` object for the given dataset.
    """
    ds: BaseDataset = module_loader(config=config, module_class="dataset", key=path)

    # validate data loader params on regular BaseDataset, a concatenated dataset should have been validated elsewhere
    if isinstance(ds, BaseDataset):
        ds.validate_params(dataloader_validations)
        params = ds.params
    else:
        assert isinstance(ds, TorchDataset)
        params = get_sub_config(config=config, path=path)

    batch_size: int = params.get("batch_size", DEF_VAL["dataloader"]["batch_size"])
    drop_last: bool = params.get("drop_last", DEF_VAL["dataloader"]["drop_last"])
    shuffle: bool = params.get("shuffle", DEF_VAL["dataloader"]["shuffle"])

    data_loader = TorchDataLoader(
        dataset=ds,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=params.get("workers", DEF_VAL["dataloader"]["workers"]),
        shuffle=shuffle,
        collate_fn=(
            collate_lists if params.get("return_lists", DEF_VAL["dataloader"]["return_lists"]) else collate_states
        ),
    )
    return data_loader
