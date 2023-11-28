"""
Module for handling data loading and preprocessing using torch Datasets.
"""
from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from dgs.models.states import BackboneOutput, BackboneOutputs


class BaseDataset(TorchDataset):
    """Custom Dataset"""

    def __init__(self, filenames: list[str]) -> None:
        # store list of filenames as np array, to reduce memory usage on multiple devices
        super().__init__()

        self.filenames: np.ndarray = np.array(filenames)

    @abstractmethod
    def __len__(self) -> int:
        """Override len() functionality for torch."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> BackboneOutput:
        """Retrieve data at index from given dataset.

        Args:
            idx: index of the data object. Is a reference to the same object as len().

        Returns:
            Precomputed backbone output
        """
        raise NotImplementedError

    def __getitems__(self, indices: list[int]) -> BackboneOutputs:
        """Get batch of data given list of indices.

        'Subclasses could also optionally implement __getitems__(), for speedup batched samples loading.
        This method accepts list of indices of samples of batch and returns list of samples'

        Args:
            indices: A list of indices to retrieve data from
        """
        raise NotImplementedError
