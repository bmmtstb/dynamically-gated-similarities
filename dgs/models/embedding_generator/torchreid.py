"""
Visual Re-ID module using the torchreid package.
"""
import torch

from dgs.models.embedding_generator.reid import EmbeddingGeneratorModule
from dgs.utils.types import Config
from dgs.utils.utils import project_to_abspath
from torchreid import models
from torchreid.models import __model_factory as torchreid_models
from torchreid.utils.torchtools import load_pretrained_weights

torchreid_validations: Config = {"model_name": ["str", ("in", torchreid_models.keys())]}


class TorchreidModel(EmbeddingGeneratorModule):
    """Given image crops, generate Re-ID embedding using the torchreid package.

    Model can use the default pretrained weights or custom weights.
    """

    def __init__(self, config, path):
        super().__init__(config=config, path=path)

        self.model_weights = self.params["weights"]
        self.is_pretrained = self.model_weights == "pretrained"
        self.init_model()

    def init_model(self) -> None:
        """Initialize torchreid model"""
        m = models.build_model(
            name=self.params["model_name"], num_classes=self.embedding_size, pretrained=self.is_pretrained
        )
        if not self.is_pretrained:
            # custom model params
            load_pretrained_weights(m, project_to_abspath(self.model_weights))

        # send model to device
        self.model = self.configure_torch_model(m)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)
