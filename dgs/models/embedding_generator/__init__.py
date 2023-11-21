"""
Modules for Re-Identification ("Re-ID").

May either be for visual Re-ID or for pose-based Re-ID.
Given some input data, predict something like an embedding vector to describe the input.
This vector is then used to compare it to other embedding vectors using a similarity function.
"""

__all__ = ["TorchreidModel"]


from dgs.models.embedding_generator.torchreid import TorchreidModel
