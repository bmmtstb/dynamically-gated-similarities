"""
functions and models for generating and handling different Re-ID models or other embedding-based models
"""
from abc import abstractmethod

import torch
from torch import nn

from dgs.models.model import BaseModule
from dgs.utils.types import Config
from torchreid import models
from torchreid.utils.torchtools import load_pretrained_weights


class EmbeddingGeneratorModel(BaseModule):
    """
    Base class for handling embedding generators.

    Given data predict an embedding describing the data.
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

        Parameters
        ----------
        data: array-like, np or torch

            input data to create embedding from
            If self.is_batched is true, shape should be [B x ...] otherwise [...]

        Returns
        -------
        array-like torch object
        """

        raise NotImplementedError


class TorchReID(EmbeddingGeneratorModel):
    """
    given image crops, generate Re-ID embedding using torchreid package
    """

    def __init__(self, config, path):
        super().__init__(config=config, path=path)

        self.model_name = self.config.model_name
        self.model_weights_path = self.config.model_weights_path
        self.is_pretraind = self.config.model_weights_path == "pretrained"
        self.init_model()

    def init_model(self) -> None:
        """initialize model for class"""
        # if self.model_name == "osnet_ain_x1_0":
        #     m = osnet_ain_x1_0(num_classes=self.embedding_size, pretrained=True)
        # else:
        #     raise NotImplementedError

        m = models.build_model(name=self.model_name, num_classes=self.embedding_size, pretrained=self.is_pretraind)
        if not self.is_pretraind:
            # custom model params
            load_pretrained_weights(m, self.model_weights_path)

        self.model = nn.DataParallel(m, device_ids=self.config.gpus).to(self.config.device).eval()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        raise NotImplementedError
