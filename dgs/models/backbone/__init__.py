"""
Backbone models will compute additional information for given States.
"""

from typing import Type

from dgs.utils.loader import get_instance, register_instance
from .backbone import BaseBackboneModule
from .yolo import YoloBackbone

BACKBONES: dict[str, Type[BaseBackboneModule]] = {
    "yolo-v7": YoloBackbone,
    # "alpha-pose": ...,
    # "AlphaPose": ...,
}


def get_backbone(name: str) -> Type[BaseBackboneModule]:
    """Given the name of one backbone module, return an instance."""
    return get_instance(instance=name, instances=BACKBONES, inst_class=BaseBackboneModule)


def register_backbone(name: str, new_back: Type[BaseBackboneModule]) -> None:
    """Register a new backbone module in :data:`BACKBONES`, to be able to use it from configuration files."""
    register_instance(name=name, instance=new_back, instances=BACKBONES, inst_class=BaseBackboneModule)
