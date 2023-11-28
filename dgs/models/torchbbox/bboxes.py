"""
Custom class for torch-based bounding boxes.

Basically is a list of BoundingBox objects with a few more functions.
"""
from collections import UserList
from typing import Iterable

from dgs.models.torchbbox.bbox import BoundingBox

BoundingBoxesIterable = Iterable[BoundingBox]


class BoundingBoxes(UserList):
    """List of multiple bounding boxes with some additional functionality"""

    def __init__(self, boxes: list[BoundingBox] = None) -> None:
        self.data: list[BoundingBox]  # will be set by super init
        super().__init__(boxes)
