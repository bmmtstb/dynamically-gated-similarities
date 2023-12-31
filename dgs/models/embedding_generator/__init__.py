"""
Modules for generating embeddings for Re-Identification ('ReID-Embeddings').

May either be for visual Re-ID or for pose-based Re-ID.
Given some input data, predict something like an embedding vector to describe the input.
This vector is then used to compare it to other embedding vectors using a similarity function.
The vector of one specified target should be close to other generated vectors of this target,
but as far as possible from independent targets.
"""
from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .embedding_generator import EmbeddingGeneratorModule
from .pose_based import KeyPointConvolutionPBEG, LinearPBEG
from .torchreid import TorchreidModel


def get_embedding_generator(name: str) -> Type[EmbeddingGeneratorModule]:
    """Given the name of one dataset, return an instance."""
    if name == "torchreid":
        return TorchreidModel
    if name == "LinearPBEG":
        return LinearPBEG
    if name == "KeyPointConvolutionPBEG":
        return KeyPointConvolutionPBEG
    raise InvalidParameterException(f"Unknown embedding generator with name: {name}.")
