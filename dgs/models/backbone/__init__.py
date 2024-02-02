"""
Different modules that can be used as a backbone to generate or retrieve data.
"""

from typing import Type

from dgs.utils.exceptions import InvalidParameterException
from .alphapose import AlphaPoseFullBackbone
from .backbone import BackboneModule


def get_backbone(name: str) -> Type[BackboneModule]:
    """Given the name of one backbone module and return an instance."""
    if name == "AlphaPose":
        return AlphaPoseFullBackbone
    raise InvalidParameterException(f"Unknown backbone with name: {name}.")
