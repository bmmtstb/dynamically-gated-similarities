"""
Base module for models used for embedding generation.
"""

from abc import abstractmethod

import torch
from torch import nn

from dgs.models.module import BaseModule
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

embedding_validations: Validations = {
    "module_name": [str],
    "embedding_size": [int, ("gt", 0)],
    "nof_classes": [int, ("gt", 0)],
}


class EmbeddingGeneratorModule(BaseModule, nn.Module):
    """Base class for handling modules of embedding generators.

    Description
    -----------

    Given some model-specific data through the State, child models of this class will predict one embedding
    per single sample (detection) of data, describing it.
    The child models should also work for batched input data.

    Params
    ------

    embedding_size (int):
        The size of the embedding.
        This size does not necessarily have to match other embedding sizes.
    nof_classes (int):
        The number of classes in the dataset.
        Used during training to predict the id.
    """

    embedding_size: int
    """The size of the embedding."""

    nof_classes: int
    """The number of classes in the dataset / embedding."""

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config=config, path=path)
        nn.Module.__init__(self)

        self.validate_params(embedding_validations)

        self.embedding_size = self.params["embedding_size"]
        self.nof_classes = self.params["nof_classes"]

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, ds: State) -> torch.Tensor:
        """Predict next outputs, given any data in a State object, using this Re-ID model.

        Returns:
            The generated embeddings as tensor of shape ``[N x embedding_size]``.
        """
        raise NotImplementedError
