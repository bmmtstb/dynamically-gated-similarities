"""
Visual Re-ID module using the torchreid package.
"""
import torch
from torch import nn

from dgs.models.reid.reid import EmbeddingGeneratorModule
from torchreid import models
from torchreid.utils.torchtools import load_pretrained_weights


class TorchreidModel(EmbeddingGeneratorModule):
    """
    given image crops, generate Re-ID embedding using torchreid package
    """

    def __init__(self, config, path):
        super().__init__(config=config, path=path)

        self.model_name = self.config.model_name
        self.model_weights_path = self.config.model_weights_path
        self.is_pretrained = self.config.model_weights_path == "pretrained"
        self.init_model()

    def init_model(self) -> None:
        """initialize model for class"""
        # if self.model_name == "osnet_ain_x1_0":
        #     m = osnet_ain_x1_0(num_classes=self.embedding_size, pretrained=True)
        # else:
        #     raise NotImplementedError

        m = models.build_model(name=self.model_name, num_classes=self.embedding_size, pretrained=self.is_pretrained)
        if not self.is_pretrained:
            # custom model params
            load_pretrained_weights(m, self.model_weights_path)

        self.model = nn.DataParallel(m, device_ids=self.config.gpus).to(self.config.device).eval()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        raise NotImplementedError
