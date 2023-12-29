"""
Modules for loading data.
The modules are a combination of my custom BaseModule and a regular torch Dataset.
"""
from typing import Type

from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.types import Config
from .alphapose import AlphaPoseLoader
from .dataset import BaseDataset, collate_data_samples
from .posetrack import PoseTrack21, PoseTrack21JSON


def get_dataset(name: str) -> Type[BaseDataset]:
    """Given the name of one dataset, return an instance."""
    # import only what is needed and reduce the chance for circular imports
    if name == "PoseTrack21":
        return PoseTrack21
    if name == "PoseTrack21JSON":
        return PoseTrack21JSON
    if name == "AlphaPoseLoader":
        return AlphaPoseLoader
    raise InvalidParameterException(f"Unknown dataset with name: {name}.")


def get_data_loader(config: Config, ds: TorchDataset) -> TorchDataLoader:
    """Set up torch data loader with some params from config.

    Args:
        config: Overall tracker configuration.
        ds: Reference to torch Dataset.

    Returns:
        A `torch.DataLoader` object for the given dataset.
    """
    data_loader = TorchDataLoader(
        dataset=ds,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        collate_fn=collate_data_samples,
    )
    # https://glassboxmedicine.com/2020/03/04/multi-gpu-training-in-pytorch-data-and-model-parallelism/
    # By default, num_workers is set to 0.
    # Setting num_workers to a positive integer turns on multiprocess data loading in which data will be loaded using
    # the specified number of loader worker processes.
    return data_loader
