"""
Compute the similarity using one of the torchreid models.
"""

import warnings

import torch
from torch import nn

from dgs.models.metric import get_metric, METRICS
from dgs.models.similarity import SimilarityModule
from dgs.utils.files import to_abspath
from dgs.utils.torchtools import load_pretrained_weights
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
            num_classes=self.params["nof_classes"],
            pretrained=pretrained,
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
        sim_kwargs = self.params.get("similarity_kwargs", {})
        m = get_metric(name)(**sim_kwargs)

        # send function to the device
        return self.configure_torch_module(m, train=False)

    def forward(self, data: torch.Tensor, target: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Forward call of the torchreid model.

        Will first compute the models' output of shape ``[a x E]``.
        Then use this modules' metric
        to compute the similarity between the predicted and the target embeddings with a shape of ``[a x b]``.

        Notes:
            Torchreid expects images to have float values.

        Args:
            data: The image crop to compute the embedding from, shape: ``[a x C x w x h]``.
            target: The target embeddings, shape: ``[b x E]``.

        Returns:
            The similarity between the predictions using this model and given targets as tensor of shape ``[a x b]``.
        """

        def _get_torchreid_results(r) -> torch.Tensor:
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
        pred_embeds = _get_torchreid_results(results)
        # pred embeds have shape [A x E]

        return self.func(pred_embeds, target)
