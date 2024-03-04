"""
Modules for generating embeddings for Re-Identification ('ReID-Embeddings').

May either be for visual Re-ID or for pose-based Re-ID.
Given some input data, predict something like an embedding vector to describe the input.
This vector is then used to compare it to other embedding vectors using a similarity function.
The vector of one specified target should be close to other generated vectors of this target,
but as far as possible from independent targets.

For using an EmbeddingGeneratorModule as similarity, see the description of :class:`~.SimilarityModule`.
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from .embedding_generator import EmbeddingGeneratorModule
from .pose_based import KeyPointConvolutionPBEG, LinearPBEG

EMBEDDING_GENERATORS: dict[str, Type[EmbeddingGeneratorModule]] = {
    "LinearPBEG": LinearPBEG,
    "KeyPointConvolutionPBEG": KeyPointConvolutionPBEG,
}


def get_embedding_generator(name: str) -> Type[EmbeddingGeneratorModule]:
    """Given the name or a new instance of an embedding generator module, return the type."""
    return get_instance(instance=name, instances=EMBEDDING_GENERATORS, inst_class=EmbeddingGeneratorModule)


def register_embedding_generator(name: str, new_eg: Type[EmbeddingGeneratorModule]) -> None:
    """Register a new embedding generator in ``EMBEDDING_GENERATORS``, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_eg, instances=EMBEDDING_GENERATORS, inst_class=EmbeddingGeneratorModule)
