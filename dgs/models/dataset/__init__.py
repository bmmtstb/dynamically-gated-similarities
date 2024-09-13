r"""
Modules for loading data, including datasets and data loaders.
The modules are a combination of my custom BaseModule and a regular torch Dataset.
Additionally, I implemented a Dataset for the |PT21|_ dataset that can be loaded within |torchreid|_.
"""

import os
from glob import glob
from typing import Type, Union

from torch.utils.data import ConcatDataset as TConcatDataset
from tqdm import tqdm

from dgs.utils.config import get_sub_config, insert_into_config
from dgs.utils.loader import get_instance, register_instance
from dgs.utils.types import Config, NodePath
from .alphapose import AlphaPoseLoader
from .dataset import BaseDataset
from .keypoint_rcnn import KeypointRCNNImageBackbone, KeypointRCNNVideoBackbone
from .MOT import MOTImage, MOTImageHistory
from .posetrack21 import PoseTrack21_BBox, PoseTrack21_Image, PoseTrack21_ImageHistory

DATASETS: dict[str, Type[BaseDataset]] = {
    "PoseTrack21_BBox": PoseTrack21_BBox,
    "PT21_BBox": PoseTrack21_BBox,  # alias
    "PoseTrack21_Image": PoseTrack21_Image,
    "PT21_Image": PoseTrack21_Image,  # alias
    "AlphaPoseLoader": AlphaPoseLoader,
    "KeypointRCNNImageBackbone": KeypointRCNNImageBackbone,
    "KeypointRCNNVideoBackbone": KeypointRCNNVideoBackbone,
    "MOTImage": MOTImage,
    "MOTI": MOTImage,  # alias
    "MOTImageHistory": MOTImageHistory,
    "MOTIH": MOTImageHistory,  # alias
}


def get_dataset(name: str) -> Type[BaseDataset]:
    """Given the name of one dataset, return an instance."""
    return get_instance(instance=name, instances=DATASETS, inst_class=BaseDataset)


def register_dataset(name: str, new_ds: Type[BaseDataset]) -> None:
    """Register a new dataset module in :data:``DATASETS``, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_ds, instances=DATASETS, inst_class=BaseDataset)


def get_multi_dataset(
    config: Config, path: NodePath, ds_name: str, concat: bool = True
) -> Union[TConcatDataset[BaseDataset], list[BaseDataset]]:
    """Create a concatenated dataset from the given configuration and path.

    Args:
        config: The overall configuration for the tracker.
        path: The path to the dataset-specific parameters.
        ds_name: The type of dataset to create as a string from all the available datasets.
        concat: Whether to concatenate the list of datasets in the end. Default: True.
    """
    # get the dataset type to instantiate it faster
    ds_type = get_dataset(name=ds_name)

    sub_cfg = get_sub_config(config=config, path=path)
    if "paths" not in sub_cfg:
        raise ValueError(f"No paths given in config. Got: {sub_cfg}")

    # get all the data paths
    paths = sub_cfg["paths"]
    if isinstance(paths, (list, tuple)):
        pass
    elif isinstance(paths, str) and "*" in paths:
        paths = glob(paths)
    elif isinstance(paths, str) and os.path.exists(paths):
        paths = [paths]
    elif isinstance(paths, str) and os.path.exists(dir_path := str(os.path.join(sub_cfg["dataset_path"], paths))):
        paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    else:
        raise ValueError(f"The given 'paths' ({paths}) is neither an iterable, a string, nor a valid file path.")

    assert len(paths) > 0, f"No paths found with paths: {paths}"
    # for every dataset, insert the right data_path into the config and initialize the datasets
    datasets = []
    for data_path in tqdm(paths, desc="Loading datasets", leave=False):
        ds_cfg = insert_into_config(
            path=path,
            value={
                "data_path": os.path.normpath(data_path),
                "module_name": str(ds_name).lstrip("Concat_"),
            },
            original=config,
            copy=True,
        )
        datasets.append(ds_type(config=ds_cfg, path=path))

    if concat:
        return TConcatDataset(datasets)
    return datasets
