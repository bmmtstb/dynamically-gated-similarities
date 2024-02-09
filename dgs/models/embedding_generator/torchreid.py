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


class TorchreidModel(EmbeddingGeneratorModule, nn.Module):
    """Given image crops, generate Re-ID embedding using the torchreid package.

    Model can use the default pretrained weights or custom weights.

    Params
    ------

    model_name (str):
        The name of the torchreid model used.
        Has to be one of ``~torchreid.models.__model_factory.keys()``.

    weights (Union[str, FilePath], optional):
        A path to the model weights or the string 'pretrained' for the default pretrained torchreid model.
        Default 'pretrained'.

    Important Inherited Params
    --------------------------

    embedding_size (int):
        The size of the embedding.
        This size does not necessarily have to match other embedding sizes.
    nof_classes (int):
        The number of classes in the dataset.
        Used during training to predict the id.

    """

    def __init__(self, config, path):
        nn.Module.__init__(self)
        EmbeddingGeneratorModule.__init__(self, config=config, path=path)

        self.model_weights = self.params.get("weights", "pretrained")
        self.is_pretrained = self.model_weights == "pretrained"
        self.model = self._init_model()

    def _init_model(self) -> nn.Module:
        """Initialize torchreid model"""
        m = build_model(
            name=self.params["model_name"],
            num_classes=self.nof_classes,
            pretrained=self.is_pretrained,
            use_gpu=self.device.type == "cuda",
            loss="triplet",  # has to be triplet for torchreid models to return embeddings and ids
        )
        if not self.is_pretrained:
            # custom model params
            load_pretrained_weights(m, to_abspath(self.model_weights))

        # send model to the device
        return self.configure_torch_module(m)

    def forward(self, *data, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward call of the torchreid model.

        Notes:
            Torchreid expects images to have float values.

        Args:
            data: The image crop to compute the embedding from.

        Returns:
            Output of this models' forward call as the predicted embeddings ``[B x E]`` and
            the predicted class probabilities ``[B x num_classes]``.
        """
        r = self.model(to_dtype(*data, dtype=torch.float32), **kwargs)

        if len(r) == 2:
            ids, embeddings = r
            return embeddings, ids
        if isinstance(r, torch.Tensor):
            # During model building, triplet loss was forced for torchreid models.
            # Therefore, only one return value means that only the embeddings are returned
            embeddings = r
            if hasattr(self.model, "classifier"):
                ids = self.model.classifier(embeddings)
                return embeddings, ids
            raise NotImplementedError("Only the embeddings are returned and there is no classifier in torchreid model.")
        raise NotImplementedError("Unknown torchreid model output.")
