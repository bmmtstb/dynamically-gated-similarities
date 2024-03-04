r"""
Modules for loading data, including datasets and data loaders.
The modules are a combination of my custom BaseModule and a regular torch Dataset.
Additionally, I implemented a Dataset for the |PT21|_ dataset that can be loaded within |torchreid|_.
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from .alphapose import AlphaPoseLoader
from .dataset import BaseDataset, collate_data_samples
from .posetrack21 import PoseTrack21JSON

DATASETS: dict[str, Type[BaseDataset]] = {
    "PoseTrack21JSON": PoseTrack21JSON,
    "AlphaPoseLoader": AlphaPoseLoader,
}


def get_dataset(name: str) -> Type[BaseDataset]:
    """Given the name of one dataset, return an instance."""
    return get_instance(instance=name, instances=DATASETS, inst_class=BaseDataset)


def register_combine_module(name: str, new_ds: Type[BaseDataset]) -> None:
    """Register a new daatset module in ``DATASETS``, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_ds, instances=DATASETS, inst_class=BaseDataset)
