r"""
Modules for loading data, including datasets and data loaders.
The modules are a combination of my custom BaseModule and a regular torch Dataset.
Additionally, I implemented a Dataset for the |PT21|_ dataset that can be loaded within |torchreid|_.
"""

from typing import Type

from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from dgs.utils.exceptions import InvalidParameterException
from .alphapose import AlphaPoseLoader
from .dataset import BaseDataset, collate_data_samples
from .posetrack21 import PoseTrack21JSON


def get_dataset(name: str) -> Type[BaseDataset]:
    """Given the name of one dataset, return an instance."""
    # import only what is needed and reduce the chance for circular imports
    if name == "PoseTrack21JSON":
        return PoseTrack21JSON
    if name == "AlphaPoseLoader":
        return AlphaPoseLoader
    raise InvalidParameterException(f"Unknown dataset with name: {name}.")


def get_data_loader(ds: TorchDataset, batch_size: int, **kwargs) -> TorchDataLoader:
    """Set up a torch data loader with some params from config.

    Args:
        ds: Reference to torch Dataset.
        batch_size: Size of the batches within this DataLoader.

    Keyword Args:
        workers (int): Number of workers for this DataLoader.
        shuffle (bool): Whether to shuffle the Dataset in this DataLoader

    Returns:
        A `torch.DataLoader` object for the given dataset.
    """
    data_loader = TorchDataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=kwargs.get("workers", 0),
        shuffle=kwargs.get("shuffle", False),
        collate_fn=collate_data_samples,
    )
    # https://glassboxmedicine.com/2020/03/04/multi-gpu-training-in-pytorch-data-and-model-parallelism/
    # By default, num_workers is set to 0.
    # Setting num_workers to a positive integer turns on multiprocess data loading in which data will be loaded using
    # the specified number of loader worker processes.
    return data_loader
