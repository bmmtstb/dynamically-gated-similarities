"""
Use a model out of the torchreid package as an embedding generator.
"""

import warnings

import torch as t
from torch import nn

from dgs.models.embedding_generator.embedding_generator import EmbeddingGeneratorModule
from dgs.utils.config import DEF_VAL, get_sub_config, insert_into_config
from dgs.utils.exceptions import InvalidPathException
from dgs.utils.files import to_abspath
from dgs.utils.state import State
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
    "image_key": ["optional", str],
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

    Notes:
        Setting the parameter ``embedding_size`` does not change this module's output.
        Torchreid does not support custom embedding sizes.

    Module Name
    -----------

    torchreid

    Params
    ------

    model_name (str):
        The name of the torchreid model used.
        Has to be one of ``~torchreid.models.__model_factory.keys()``.

    Optional Params
    ---------------

    weights (Union[str, FilePath], optional):
        A path to the model weights or the string 'pretrained' for the default pretrained torchreid model.
        Default ``DEF_VAL.embed_gen.torchreid.weights``.
    image_key (str, optional):
        The key of the image to use when generating the embedding.
        Default ``DEF_VAL.embed_gen.torchreid.image_key``.

    Important Inherited Params
    --------------------------

    nof_classes (int):
        The number of classes in the dataset.
        Used during training to predict the class-id.
        For most of the pretrained torchreid models, this ist set to ``1_000``.

    """

    model: nn.Module

    def __init__(self, config, path):
        if path is None:
            raise InvalidPathException("path is required but got None")
        sub_cfg = get_sub_config(config, path)
        if "embedding_size" in sub_cfg and sub_cfg["embedding_size"] != 512:
            warnings.warn(
                "The embedding size will be overwritten in torchreid embedding generators, "
                "because torchreid does not support different sizes."
            )
        new_cfg = insert_into_config(path=path, value={"embedding_size": 512}, original=config)
        del config

        EmbeddingGeneratorModule.__init__(self, config=new_cfg, path=path)

        self.model_weights = self.params.get("weights", DEF_VAL["embed_gen"]["torchreid"]["weights"])

        model = self._init_model(self.model_weights == "pretrained")
        self.register_module(name="model", module=self.configure_torch_module(model))

        self.image_key = self.params.get("image_key", DEF_VAL["embed_gen"]["torchreid"]["image_key"])

    def _init_model(self, pretrained: bool) -> nn.Module:
        """Initialize torchreid model"""
        m = build_model(
            name=self.params["model_name"],
            num_classes=self.params["nof_classes"],
            pretrained=pretrained,
            use_gpu=self.device.type == "cuda",
            loss="triplet",
        )
        if not pretrained:  # pragma: no cover
            # custom model params
            load_pretrained_weights(m, to_abspath(self.model_weights))
        return m

    def predict_embeddings(self, data: t.Tensor) -> t.Tensor:
        """Predict embeddings given some input.

        Args:
            data: The input for the model, most likely a cropped image.

        Returns:
            Tensor containing a batch B of embeddings.
            Shape: ``[B x E]``
        """

        def _get_torchreid_embeds(r) -> t.Tensor:
            """Torchreid returns embeddings during eval and ids during training."""
            if isinstance(r, t.Tensor):
                # During model building, triplet loss was forced for torchreid models.
                # Therefore, only one return value means that only the embeddings are returned
                return r
            if len(r) == 2:
                _, embeddings = r
                return embeddings
            raise NotImplementedError("Unknown torchreid model output.")

        results = self.model(data)
        return _get_torchreid_embeds(results)

    def predict_ids(self, data: t.Tensor) -> t.Tensor:
        """Predict class IDs given some input.

        Args:
            data: The input for the model, most likely a cropped image.

        Returns:
            Tensor containing class predictions, which are not necessarily a probability distribution.
            Shape: ``[B x num_classes]``
        """

        def _get_torchreid_ids(r) -> t.Tensor:
            """Torchreid returns embeddings during eval and ids during training."""
            if isinstance(r, t.Tensor):
                # During model building, triplet loss was forced for torchreid models.
                # Therefore, only one return value means that only the embeddings are returned
                return self.model.classifier(r)
            if len(r) == 2:
                ids, _ = r
                return ids
            raise NotImplementedError("Unknown torchreid model output.")

        results = self.model(data)
        return _get_torchreid_ids(results)

    def forward(self, ds: State) -> t.Tensor:
        """Predict embeddings given some input.

        Notes:
            Torchreid models will return different results based on whether they are in eval or training mode.
            Make sure forward is only called in the evaluation mode.

        Args:
            ds: A :class:`State` containing the cropped image as input for the model.
                :class:`Image` or FloatTensor of shape ``[B x C x w x h]``.

        Returns:
            A batch of embeddings as tensor of shape: ``[B x E]``.
        """
        if self.embedding_key_exists(ds):
            return ds[self.embedding_key]
        return self.model(getattr(ds, self.image_key) if hasattr(ds, self.image_key) else ds[self.image_key])
