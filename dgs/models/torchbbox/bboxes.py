"""
Custom class for torch-based bounding boxes.

Basically is a list of BoundingBox objects with a few more functions.
"""
from collections import UserList
from typing import Iterable

import torch

from dgs.models.torchbbox.bbox import BoundingBox

BoundingBoxesIterable = Iterable[BoundingBox]


class BoundingBoxes(UserList):
    """List of multiple bounding boxes with some additional functionality"""

    def __init__(self, boxes: list[BoundingBox] = None) -> None:
        self.data: list[BoundingBox]  # will be set by super init
        super().__init__(boxes)

    def __contains__(self, item: BoundingBox) -> bool:
        raise NotImplementedError()

    def corners(self, image_sizes: Iterable) -> torch.IntTensor:
        """Obtain the integer coordinates of the four bbox corners for every bbox.

        Args:
            image_sizes: ...?
            Support different img sizes?

        Returns:
            output shape [len x 4] - integer values

            Integer values of the four bbox corners for every bbox in self.
        """
        raise NotImplementedError()

    def iou(self) -> torch.FloatTensor:
        """Compute intersection over union between every pair of bboxes in self."""
        raise NotImplementedError()

    @property
    def xyxy(self) -> torch.FloatTensor:
        """Get all the bboxes in xyxy format"""
        raise NotImplementedError()

    @property
    def xywh(self) -> torch.FloatTensor:
        """Get all the bboxes in xywh format"""
        raise NotImplementedError()

    @property
    def xyah(self) -> torch.FloatTensor:
        """Get all the bboxes in xyah format"""
        raise NotImplementedError()

    @property
    def yolo(self) -> torch.FloatTensor:
        """Get all the bboxes in yolo format"""
        raise NotImplementedError()
