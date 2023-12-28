r"""Functions and models for generating and handling different Re-ID models or other embedding-based models.

Embedding Shapes
================

The embedding shapes do have an influence on the performance and should be chosen carefully.
With that said, it is possible to have different embedding shapes throughout the tracker,
as long as the shapes match when inserting them into the similarity models.
(Or there is a similarity model, that supports different shaped inputs...)


The default embedding generators can be separated into two main classes:

Visual Re-ID embedding generators:
----------------------------------

Given image data, these models predict a single vector describing the image.
How the model does that depends on the specific model.
A fairly straightforward example might use a CNN with a fully connected output layer.

Some examples can be seen in the torchreid package within the
_`Re-ID-specific models <torchreid models>` section.

Pose-based embedding generators:
--------------------------------

Given pose information (joint coordinates, bbox information, ...) predict a single vector describing the
detected person.

As long as the Pose-based embedding doesn't have other constraints on its size,
it is wise to have an embedding size smaller than the overall initial input size.
Otherwise, the model can just append random noise to the inputs and use the plain inputs as outputs,
contradicting the goal of generating useful information.

For N detections, or a batch size of N, the output size will be [N x E :sub:´P´].
The inputs have the following shapes (first w.o. velocities, second with):

- pose: ``[N x J x 2]`` and ``[N x J x 4]``
- bounding-box: ``[N x 4]`` and ``[N x 8]``
- joint-confidence score, visibility or similar: ``[N x J x 1]``

Therefore, without velocity scores, there are :math:`N \cdot (J \cdot 3 + 4)` input values.
This means that for :math:`E_P \lt J \cdot 3 + 4` the amount of parameters gets reduced in the model.

With velocity scores, a parameter reduction is obtained with :math:`E_P \lt J \cdot 5 + 8`.

:: _torchreid models: https://kaiyangzhou.github.io/deep-person-reid/pkg/models.html#reid-specific-models
"""
from abc import abstractmethod

import torch

from dgs.models.module import BaseModule
from dgs.utils.types import Config, NodePath, Validations

embedding_validations: Validations = {"embedding_size": ["int", ("gt", 0)]}


class EmbeddingGeneratorModule(BaseModule):
    """Base class for handling modules of embedding generators.

    Description
    -----------

    Given some model-specific data, child models will predict one embedding,
    per single sample of data, describing it.
    The child models should also work for batched input data.

    Params
    ------

    embedding_size: (int)
        The size of the embedding.
        It does not necessarily have to match other embedding sizes.
    """

    embedding_size: int
    """The size of the embedding. It Does not necessarily have to match other embedding sizes."""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)
        self.validate_params(embedding_validations)

        self.embedding_size = self.params["embedding_size"]

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Predict next outputs using this Re-ID model.

        Returns:
            The generated embedding with a shape of ``[N x E]``, with E being model-dependent.
        """

        raise NotImplementedError
