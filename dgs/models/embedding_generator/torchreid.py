"""
Visual Re-ID module using the torchreid package.
"""
import warnings

import torch
from torch import nn
from torchvision.transforms.v2.functional import to_dtype

from dgs.models.embedding_generator.embedding_generator import EmbeddingGeneratorModule
from dgs.utils.files import to_abspath
from dgs.utils.types import Config

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Cython evaluation.*is unavailable", category=UserWarning)
    # ignore cython warning
    try:
        # If torchreid is installed using `./dependencies/torchreid`
        # noinspection PyUnresolvedReferences LongLine
        from torchreid.models import __model_factory as torchreid_models, build_model

        # noinspection PyUnresolvedReferences LongLine
        from torchreid.utils.torchtools import load_pretrained_weights
    except ModuleNotFoundError:
        # if torchreid is installed using `pip install torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.reid.models import __model_factory as torchreid_models, build_model

        # noinspection PyUnresolvedReferences
        from torchreid.reid.utils.torchtools import load_pretrained_weights

torchreid_validations: Config = {"model_name": ["str", ("in", torchreid_models.keys())]}


class TorchreidModel(EmbeddingGeneratorModule):
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
        m = build_model(
            name=self.params["model_name"],
            num_classes=self.embedding_size,
            pretrained=self.is_pretrained,
            use_gpu=self.device.type == "cuda",
        )
        if not self.is_pretrained:
            # custom model params
            load_pretrained_weights(m, to_abspath(self.model_weights))

        # send model to the device
        return self.configure_torch_module(m)

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
