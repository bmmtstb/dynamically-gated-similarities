"""
Use a model out of the torchreid package as an embedding generator.
"""

import warnings

import torch
from torch import nn

from dgs.models.embedding_generator.embedding_generator import EmbeddingGeneratorModule
from dgs.utils.files import to_abspath
from dgs.utils.states import DataSample
from dgs.utils.torchtools import configure_torch_module, load_pretrained_weights
from dgs.utils.types import Config

with warnings.catch_warnings():
    # ignore cython warning
    warnings.filterwarnings("ignore", message="Cython evaluation.*is unavailable", category=UserWarning)
    try:
        # If torchreid is installed using `./dependencies/torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.models import __model_factory as torchreid_models, build_model
    except ModuleNotFoundError:
        # if torchreid is installed using `pip install torchreid`
        # noinspection PyUnresolvedReferences
        from torchreid.reid.models import __model_factory as torchreid_models, build_model

torchreid_validations: Config = {
    "model_name": [str, ("in", torchreid_models.keys())],
    # optional
    "weights": [
        "optional",
        (
            "or",
            [("eq", "pretrained"), "file exists", "file exists in project", ("file exists in folder", "./weights/")],
        ),
    ],
}


@configure_torch_module
class TorchreidEmbeddingGenerator(EmbeddingGeneratorModule):
    """Given image crops, generate embedding using the torchreid package.

    The model can use the default pretrained weights or custom weights.

    Notes:
        This model will be set to evaluate only right now!
        Pretrain your models using the |torchreid| package and possibly the custom PT21 data loaders,
        then load the weights.
        The classifier is not required for embedding generation.

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

    model: nn.Module

    def __init__(self, config, path):
        EmbeddingGeneratorModule.__init__(self, config=config, path=path)

        self.model_weights = self.params.get("weights", "pretrained")

        self.model = self._init_model(self.model_weights == "pretrained")
        self.add_module(name="model", module=self.model)

    def _init_model(self, pretrained: bool) -> nn.Module:
        """Initialize torchreid model"""
        m = build_model(
            name=self.params["model_name"],
            num_classes=self.params["nof_classes"],
            pretrained=pretrained,
            use_gpu=self.device.type == "cuda",
        )
        if not pretrained:
            # custom model params
            load_pretrained_weights(m, to_abspath(self.model_weights))
        # send model to the device
        return self.configure_torch_module(m, train=False)

    def forward(self, ds: DataSample) -> torch.Tensor:
        """Predict embeddings given some input.

        Notes:
            Torchreid models will return different results based on whether they are in eval or training mode.
            Make sure forward is only called in the evaluation mode.

        Args:
            ds: A DataSample containing the cropped image as input for the model.
                TVImage or FloatTensor of shape ``[B x C x w x h]``.

        Returns:
            A batch of embeddings as tensor of shape: ``[B x E]``.
        """
        if "embedding" in ds:
            return ds["embedding"]
        return self.model(ds.image_crop)
