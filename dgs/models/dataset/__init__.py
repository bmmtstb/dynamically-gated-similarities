r"""
Modules for loading data, including datasets and data loaders.
The modules are a combination of my custom BaseModule and a regular torch Dataset.
Additionally, I implemented a Dataset for the |PT21|_ dataset that can be loaded within |torchreid|_.
"""

from typing import Type

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
