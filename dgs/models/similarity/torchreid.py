"""
Compute the similarity using one of the torchreid models.
"""

import warnings

import torch
from torch import nn

from dgs.models.metric import get_metric, METRICS
from dgs.models.similarity.similarity import SimilarityModule
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
    "similarity": [str, ("in", METRICS.keys())],
    # optional
    "weights": [
        "optional",
        (
            "or",
            [("eq", "pretrained"), "file exists", "file exists in project", ("file exists in folder", "./weights/")],
        ),
    ],
    "similarity_kwargs": ["optional", dict],
}


@configure_torch_module
class TorchreidSimilarity(SimilarityModule):
    """Given image crops, generate Re-ID embedding using the torchreid package.

    Model can use the default pretrained weights or custom weights.

    Notes:
        This model cannot be trained right now!
        Pretrain your models using the |torchreid| package and possibly the custom PT21 data loaders,
        then load the weights.

    Notes:
        For computing the similarity during evaluation,
        most models should re-use the distance function used during training.

    Params
    ------

    model_name (str):
        The name of the torchreid model used.
        Has to be one of ``~torchreid.models.__model_factory.keys()``.
    similarity (str):
        The name of the similarity function / metric to use.
        Has to be one of ``dgs.models.metric.METRICS``

    weights (Union[str, FilePath], optional):
        A path to the model weights or the string 'pretrained' for the default pretrained torchreid model.
        Default 'pretrained'.
    similarity_kwargs (dict, optional):
        Possibly pass additional kwargs to the similarity function.
        Default {}.
    nof_classes (int, optional):
        The number of classes in the dataset.
        Should only be set if the classes are used at any point.
        Default 1000.
    """

    model: nn.Module
    func: nn.Module

    def __init__(self, config, path):
        SimilarityModule.__init__(self, config=config, path=path)

        self.model_weights = self.params.get("weights", "pretrained")

        self.model = self._init_model(self.model_weights == "pretrained")
        self.add_module(name="model", module=self.model)

        self.func = self._init_func()
        self.add_module(name="func", module=self.func)

    def _init_model(self, pretrained: bool) -> nn.Module:
        """Initialize torchreid model"""
        m = build_model(
            name=self.params["model_name"],
            num_classes=self.params.get("nof_classes", 1000),
            pretrained=pretrained,
            loss="triplet",  # we always want to use the embeddings!
            use_gpu=self.device.type == "cuda",
        )
        if not pretrained:
            # custom model params
            load_pretrained_weights(m, to_abspath(self.model_weights))
        # send model to the device
        return self.configure_torch_module(m, train=False)

    def _init_func(self) -> nn.Module:
        """Initialize the similarity function"""
        name = self.params["similarity"]
        m = get_metric(name)(**self.params.get("similarity_kwargs", {}))

        # send function to the device
        return self.configure_torch_module(m, train=False)

    def predict_ids(self, data: torch.Tensor) -> torch.Tensor:
        """Predict class IDs given some input.

        Args:
            data: The input for the model, most likely a cropped image.

        Returns:
            Tensor containing class predictions, which are not necessarily a probability distribution.
            Shape: ``[B x num_classes]``
        """

        def _get_torchreid_ids(r) -> torch.Tensor:
            """Torchreid returns embeddings during eval and ids during training."""
            if isinstance(r, torch.Tensor):
                # During model building, triplet loss was forced for torchreid models.
                # Therefore, only one return value means that only the embeddings are returned
                return self.model.classifier(r)
            if len(r) == 2:
                ids, _ = r
                return ids
            raise NotImplementedError("Unknown torchreid model output.")

        results = self.model(data)
        return _get_torchreid_ids(results)

    def predict_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        """Predict embeddings given some input.

        Args:
            data: The input for the model, most likely a cropped image.

        Returns:
            Tensor containing a batch B of embeddings.
            Shape: ``[B x E]``
        """

        def _get_torchreid_embeds(r) -> torch.Tensor:
            """Torchreid returns embeddings during eval and ids during training."""
            if isinstance(r, torch.Tensor):
                # During model building, triplet loss was forced for torchreid models.
                # Therefore, only one return value means that only the embeddings are returned
                return r
            if len(r) == 2:
                _, embeddings = r
                return embeddings
            raise NotImplementedError("Unknown torchreid model output.")

        results = self.model(data)
        return _get_torchreid_embeds(results)

    def get_data(self, ds: DataSample) -> torch.Tensor:
        """Given a DataSample get the current embedding or compute it using the image crop."""
        if "embedding" in ds:
            return ds["embedding"]
        return self.model(ds.image_crop)

    def get_target(self, ds: DataSample) -> torch.Tensor:
        """Given a DataSample get the target embedding or compute it using the image crop."""
        if "embedding" in ds:
            return ds["embedding"]
        return self.model(ds.image_crop)

    def forward(self, data: DataSample, target: DataSample) -> torch.Tensor:
        """Forward call of the torchreid model used to compute the similarities between visual embeddings.

        Either load or compute the visual embeddings for the data and target using the model.
        The embeddings are tensors of respective shapes ``[a x E]`` and ``[b x E]``.
        Then use this modules' metric to compute the similarity between the two embeddings.

        Notes:
            Torchreid expects images to have float values.

        Args:
            data: A DataSample containing the predicted embedding or the image crop.
                If a predicted embedding exists, it should be stored as 'embedding' in the DataSample.
                ``self.get_data()`` will then extract the embedding as tensor of shape: ``[a x E]``.
            target: A Data Sample containing either the target embedding or the image crop.
                If a predicted embedding exists, it should be stored as 'embedding' in the DataSample.
                ``self.get_target()`` is then used to extract embedding as tensor of shape ``[b x E]``.

        Returns:
            A similarity matrix containing values describing the similarity between every current- and target-embedding.
            The similarity is a (Float)Tensor of shape ``[a x b]`` with values in ``[0..1]``.
        """
        # pred embeds have shape [A x E]
        pred_embeds = self.get_data(ds=data)
        targ_embeds = self.get_target(ds=target)

        return self.func(pred_embeds, targ_embeds)
