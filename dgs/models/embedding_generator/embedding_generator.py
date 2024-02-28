r"""See description of `~dgs.models.similarity.similarity`."""

from abc import abstractmethod

import torch

from dgs.models.module import BaseModule
from dgs.utils.types import Config, NodePath, Validations

embedding_validations: Validations = {"embedding_size": [int, ("gt", 0)]}


class EmbeddingGeneratorModule(BaseModule):
    """Base class for handling modules of embedding generators.

    Description
    -----------

    Given some model-specific data, child models will predict one embedding,
    per single sample of data, describing it.
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
    """The size of the embedding. It Does not necessarily have to match the size of other (different) embeddings."""

    nof_classes: int
    """The number of classes in the dataset / embedding."""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)
        self.validate_params(embedding_validations)

        self.embedding_size = self.params["embedding_size"]
        self.nof_classes = self.params["nof_classes"]

    def __call__(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next outputs using this Re-ID model.

        Returns:
            The generated embedding with a shape of ``[N x E]``, with E being model-dependent.
        """

        raise NotImplementedError
