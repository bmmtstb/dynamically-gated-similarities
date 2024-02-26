"""Abstraction for modules that compute the similarity between two input tensors.

Similarity
==========

In the end, a similarity function computes a similarity score "likeness" between two inputs.
The inputs should contain at least one matching dimension.
In most cases, this is the last dimension.

Some simple pose similarities, like the "intersection over union" (IoU) ir the "object keypoint similarity" (OKS) are
computed directly on the data.
Other similarities, like most visual-based similarities,
will first compute an embedding-like representation of the input data.

Similarity Through Embeddings
=============================

An embedding is a vector that represents a trained class, e.g. a specific person.
If you have never worked with embeddings,
think of visual embeddings as a vector containing hundreds of values describing the visual appearance of this person.
E.g., the color of their clothes, skin, beard, their size, ...
But you will never know exactly what the embedding means.

More importantly,
every embedding should be close to all other embeddings of the same class (the same person)
while being as far away as possible from all the other classes (people).

The metric used to compute the likeness / distance between two embeddings does also influence the results.
Common metrics include the Euclidean distance or the cosine distance.

Embedding Shapes
----------------

The shape of an embedding is the dimensionality of the underlying state-space
(the number of values used to describe the class).
This shape largely influences the model-performance and should be chosen carefully.
With that said, it is possible to have different embedding shapes throughout the tracker,
as long as the shapes match when inserting them into the similarity models.
(Or there is a similarity model that supports different shaped inputs...)

The embedding size should always be smaller than the number of input variables in the embedding generator.
This means that an embedding size of 512 would be fine for an image of size 256x256,
but way to large for key-point coordinates of shape 21x2.
There will be an example for key-point-based embedding-generation later-on.


Embedding Generators
--------------------

The default embedding generators of this package can be separated into two main classes:

Visual Re-ID Embedding Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given image data, these models predict a single vector describing the image.
How the model does that depends on the specific model.
A fairly straightforward example might use a CNN with a fully connected output layer.

Some examples can be seen in the torchreid package within the
_`Re-ID-specific models <torchreid models>` section.

Pose-Based Embedding Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
from torch import nn

from dgs.models.module import BaseModule
from dgs.utils.types import Config, NodePath


class SimilarityModule(BaseModule, nn.Module):
    """Abstract class for similarity functions."""

    def __init__(self, config: Config, path: NodePath):
        BaseModule.__init__(self, config, path)
        nn.Module.__init__(self)

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        """Compute the similarity between two input tensors."""
        raise NotImplementedError
