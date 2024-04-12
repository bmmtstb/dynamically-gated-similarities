"""
Compute the similarity using one of the torchreid models.
"""

import torch
from torch import nn

from dgs.models.embedding_generator import TorchreidEmbeddingGenerator
from dgs.models.metric import get_metric, METRICS
from dgs.models.similarity.similarity import SimilarityModule
from dgs.utils.config import DEF_CONF
from dgs.utils.state import State
from dgs.utils.torchtools import configure_torch_module
from dgs.utils.types import Config

torchreid_validations: Config = {
    "metric": [str, ("in", METRICS.keys())],
    "embedding_generator_path": [list, ("forall", str)],
    # optional
    "softmax": ["optional", bool],
    "metric_kwargs": ["optional", dict],
}


@configure_torch_module
class TorchreidVisualSimilarity(SimilarityModule):
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

    metric (str):
        The name of the metric to use.
        Has to be one of :data:`~dgs.models.metric.METRICS`
    embedding_generator_path (:obj:`Path`):
        The path to the configuration of the embedding generator within the config.

    Optional Params
    ---------------

    metric_kwargs (dict, optional):
        Possibly pass additional kwargs to the similarity function.
        Default {}.
    softmax (bool, optional):
        Whether to compute the softmax of the similarity as the last step of the model.
        Some metrics do not return a probability distribution or even values in range :math:`[0, 1]`,
        which makes it hard to sum them up using the :class:`~.CombineSimilaritiesModule` modules.
        For example, the cosine similarity returns values in range :math:`[-1, 1]`,
        where 1 means that the vectors are close.
        On the other hand, the Euclidean distance will be larger for values that are further apart, with no upper limit.
        You can use the :class:`~.NegativeSoftmaxEuclideanDistance` or
        :class:`~.NegativeSoftmaxEuclideanSquaredDistance` as metric.
        Default `DEF_CONF.similarity.torchreid.softmax`.
    """

    model: TorchreidEmbeddingGenerator
    func: nn.Module

    def __init__(self, config, path):
        SimilarityModule.__init__(self, config=config, path=path)

        model = TorchreidEmbeddingGenerator(config=config, path=self.params.get("embedding_generator_path"))
        model.eval()
        self.add_module(name="model", module=model)

        func = self._init_func()
        self.add_module(name="func", module=func)

        self.final = nn.Sequential()
        if self.params.get("softmax", DEF_CONF.similarity.torchreid.softmax):
            self.final.append(nn.Softmax(dim=-1))

    def _init_func(self) -> nn.Module:
        """Initialize the similarity function"""
        name = self.params["metric"]
        m = get_metric(name)(**self.params.get("similarity_kwargs", {}))

        # send function to the device
        return self.configure_torch_module(m, train=False)

    def get_data(self, ds: State) -> torch.Tensor:
        """Given a :class:`State` get the current embedding or compute it using the image crop."""
        if "embedding" in ds:
            return ds["embedding"]
        ds["embedding"] = self.model.predict_embeddings(ds.image_crop)
        return ds["embedding"]

    def get_target(self, ds: State) -> torch.Tensor:
        """Given a :class:`State` get the target embedding or compute it using the image crop."""
        if "embedding" in ds:
            return ds["embedding"]
        ds["embedding"] = self.model.predict_embeddings(ds.image_crop)
        return ds["embedding"]

    def forward(self, data: State, target: State) -> torch.Tensor:
        """Forward call of the torchreid model used to compute the similarities between visual embeddings.

        Either load or compute the visual embeddings for the data and target using the model.
        The embeddings are tensors of respective shapes ``[a x E]`` and ``[b x E]``.
        Then use this modules' metric to compute the similarity between the two embeddings.

        Notes:
            Torchreid expects images to have float values.

        Args:
            data: A :class:`State` containing the predicted embedding or the image crop.
                If a predicted embedding exists, it should be stored as 'embedding' in the State.
                ``self.get_data()`` will then extract the embedding as tensor of shape: ``[a x E]``.
            target: A :class:`State` containing either the target embedding or the image crop.
                If a predicted embedding exists, it should be stored as 'embedding' in the State.
                ``self.get_target()`` is then used to extract embedding as tensor of shape ``[b x E]``.

        Returns:
            A similarity matrix containing values describing the similarity between every current- and target-embedding.
            The similarity should be (Float)Tensor of shape ``[a x b]`` with values in ``[0..1]``.
            If the provided metric does not return a probability distribution,
            you might want to change the metric or set the 'softmax' parameter of this module,
            or within the :class:`DGSModule` if this is a submodule.
            Computing the softmax ensures better / correct behavior when combining this similarity with others.
        """
        pred_embeds = self.get_data(ds=data)
        targ_embeds = self.get_target(ds=target)
        assert "embedding" in data.data, "embedding of data should be saved"
        assert "embedding" in target.data, "embedding of target should be saved"

        dist = self.func(pred_embeds, targ_embeds)

        return self.final(dist)
