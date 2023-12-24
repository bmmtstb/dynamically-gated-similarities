"""
Visual Re-ID module using the torchreid package.
"""
import torch
from torch import nn
from torchvision.transforms.v2.functional import to_dtype

from dgs.models.embedding_generator.embedding_generator import TorchEmbeddingGeneratorModule
from dgs.utils.files import to_abspath
from dgs.utils.types import Config
from torchreid import models
from torchreid.models import __model_factory as torchreid_models
from torchreid.utils.torchtools import load_pretrained_weights

torchreid_validations: Config = {"model_name": ["str", ("in", torchreid_models.keys())]}


class TorchreidModel(TorchEmbeddingGeneratorModule):
    """Given image crops, generate Re-ID embedding using the torchreid package.

    Model can use the default pretrained weights or custom weights.
    """

    def __init__(self, config, path):
        super().__init__(config=config, path=path)

        self.model_weights = self.params["weights"]
        self.is_pretrained = self.model_weights == "pretrained"
        self.model = self._init_model()

    def _init_model(self) -> nn.Module:
        """Initialize torchreid model"""
        m = models.build_model(
            name=self.params["model_name"],
            num_classes=self.embedding_size,
            pretrained=self.is_pretrained,
            use_gpu=self.device.type == "cuda",
        )
        if not self.is_pretrained:
            # custom model params
            load_pretrained_weights(m, to_abspath(self.model_weights))

        # send model to the device
        return self.configure_torch_model(m)

    def forward(self, *data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward call of the torchreid model.

        Notes:
            Torchreid expects images to have float values.

        Args:
            data: The object to compute the embedding from.

        Returns:
            Output of this models' forward call.
        """
        return self.model(to_dtype(*data, dtype=torch.float32), **kwargs)
