"""
functions and models for generating and handling different Re-ID models or other embedding-based models
"""
from abc import abstractmethod

import torch

from dgs.models.module import BaseModule
from dgs.utils.types import Config


class EmbeddingGeneratorModule(BaseModule):
    """
    Base class for handling modules of embedding generators.

    Given data predict an embedding describing the data.

    The default embedding generators can be separated into two main classes:

        Visual Re-ID embedding generators

            Given image data, these models predict a single vector describing the image.
            How the model does that depends on the specific model.
            A fairly straightforward example might use a CNN with a fully connected output layer.

        Pose-based embedding generators

            Given pose information (joint coordinates, bbox information, ...) predict a single vector describing the
            detected person.
    """

    def __init__(self, config: Config, path: list[str]):
        super().__init__(config, path)

        self.embedding_size: int = self.params.embedding_size
        self.model = None

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predict next outputs using this Re-ID model.

        Args:
            data: torch tensor
                Input data to create an embedding from.
                If self.is_batched is true, batch should be the first dimension.

        Returns:
            Next prediction
        """

        raise NotImplementedError
