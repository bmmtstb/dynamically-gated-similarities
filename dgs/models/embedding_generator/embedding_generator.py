"""
Base module for models used for embedding generation.
"""

from abc import abstractmethod

import torch as t
from torch import nn

from dgs.models.module import BaseModule, enable_keyboard_interrupt
from dgs.utils.config import DEF_VAL
from dgs.utils.state import State
from dgs.utils.types import Config, NodePath, Validations

embedding_validations: Validations = {
    "module_name": [str],
    "embedding_size": [int, ("gt", 0)],
    "nof_classes": [int, ("gt", 0)],
    # optional
    "embedding_key": ["optional", str],
    "save_embeddings": ["optional", bool],
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

    Optional Params
    ---------------

    embedding_key (str, optional):
        The key to use to retrieve the embedding of the image.
        Default ``DEF_VAL.embed_gen.embedding_key``.
    save_embeddings (bool, optional):
        Whether to save the computed embeddings in the given :class:`State`.
        Default ``DEF_VAL.embed_gen.save_embeddings``.

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
        self.embedding_key: str = self.params.get("embedding_key", DEF_VAL["embed_gen"]["embedding_key"])
        self.save_embeddings: bool = self.params.get("save_embeddings", DEF_VAL["embed_gen"]["save_embeddings"])

    def __call__(self, *args, **kwargs) -> t.Tensor:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    @enable_keyboard_interrupt
    def forward(self, ds: State) -> t.Tensor:
        """Predict next outputs, given any data in a State object, using this Re-ID model.

        Returns:
            The generated embeddings as tensor of shape ``[N x embedding_size]``.
        """
        raise NotImplementedError

    def embedding_key_exists(self, s: State) -> bool:
        """Return whether the embedding_key of this model exists in a given state."""
        return self.embedding_key in s
