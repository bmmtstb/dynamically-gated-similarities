"""
Definition of modules, loading, and further module utility functions.
"""

from copy import deepcopy
from typing import Type, TypeVar, Union

from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from dgs.models.dataset import collate_data_samples, get_dataset
from dgs.models.dataset.posetrack21 import get_pose_track_21
from dgs.models.embedding_generator import get_embedding_generator
from dgs.models.engine import get_engine
from dgs.models.module import BaseModule
from dgs.models.similarity import get_combined_similarity_module, get_similarity_module
from dgs.utils.config import get_sub_config
from dgs.utils.types import Config, NodePath

M = TypeVar("M", bound=BaseModule)


def module_loader(config: Config, module: str) -> Union[M, TorchDataLoader]:
    """Load a given module and pass down the configuration

    Args:
        config: The configuration of the current tracker

        module: Name of submodule to load

    Returns:
        Initialized instance of the submodule with its config.
    """
    # This model will have one branch for every module
    # pylint: disable=too-many-branches
    path: NodePath = [module]
    module_name: str = get_sub_config(config, path)["module_name"]

    m: Type[M]

    # Module import and initialization
    if module == "engine":
        m = get_engine(module_name)
        path = []
    elif module == "weighted_similarity":
        m = get_combined_similarity_module(module_name)
    elif module.startswith("embedding_generator_"):
        m = get_embedding_generator(module_name)
    elif module.startswith("similarity_"):
        m = get_similarity_module(module_name)
    elif module.startswith("dataloader_"):
        return get_data_loader(config=config, path=path, dl_module=module)
    elif module.startswith("dataset_"):
        # special case: the concatenated PT21 dataset is loaded via function not class
        if module_name == "PoseTrack21":
            return get_pose_track_21(config=config, path=path)
        m = get_dataset(module_name)
    else:
        raise NotImplementedError(f"Something went wrong while loading the module '{module}'")

    # instantiate module with its configuration and path
    return m(config=config, path=path)


def get_data_loader(config: Config, path: NodePath, dl_module: str) -> TorchDataLoader:
    """Set up a torch data loader with some params from config.

    Args:
        config: The overall configuration of the algorithm.
        path: The node path to the params of this DataLoader.
        dl_module: Name of the DataLoader module in the config.

    Returns:
        A `torch.DataLoader` object for the given dataset.
    """
    params = get_sub_config(config, path)

    ds_module = dl_module.replace("dataloader_", "dataset_")
    ds_config: Config = deepcopy(config)
    ds_config[ds_module] = ds_config[dl_module]

    ds: TorchDataset = module_loader(config=ds_config, module_name=ds_module)

    data_loader = TorchDataLoader(
        dataset=ds,
        batch_size=params.get("batch_size", 16),
        num_workers=params.get("workers", 0),
        shuffle=params.get("shuffle", False),
        collate_fn=collate_data_samples,
    )
    # https://glassboxmedicine.com/2020/03/04/multi-gpu-training-in-pytorch-data-and-model-parallelism/
    # By default, num_workers is set to 0.
    # Setting num_workers to a positive integer turns on multiprocess data loading in which data will be loaded using
    # the specified number of loader worker processes.
    return data_loader
