"""
Helper methods for tests.
"""
import os
from typing import Iterable

import torch
from torchvision.transforms.v2 import Compose
from torchvision.tv_tensors import BoundingBoxes

from dgs.utils.image import CustomResize, CustomToAspect, load_image
from dgs.utils.types import Device, TVImage
from dgs.utils.validation import validate_images


def test_multiple_devices(func: callable) -> callable:
    """
    Decorator to run given test on all devices.
    Check whether CUDA is available and if not use CPU only.

    Args:
        func: The decorated function

    Returns:
        The decorated function with an additional 'device' keyword-argument.
    """
    devices: list[Device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    def device_wrapper(cls, *args, **kwargs):
        for device in devices:
            func(cls, device=device, *args, **kwargs)

    return device_wrapper


def load_test_image(filename: str) -> TVImage:
    """Given the filename of an image in tests/test_data folder, load, validate and return it."""
    return validate_images(load_image(os.path.join("./tests/test_data/", filename)))


def load_test_images(filenames: Iterable[str], force_reshape: bool = False, **kwargs) -> TVImage:
    """Given the filename of an image in tests/test_data folder, load, validate and return it."""
    images = [load_image(os.path.join("./tests/test_data/", fn)) for fn in filenames]
    if force_reshape:
        transform = Compose([CustomToAspect(), CustomResize()])
        new_images = []
        for img in images:
            kw_copy = kwargs.copy()
            data = {
                "image": img.detach().clone(),
                "box": kw_copy.pop("box", BoundingBoxes(torch.zeros((1, 4)), format="XYWH", canvas_size=(1, 1))),
                "keypoints": kw_copy.pop("keypoints", torch.zeros((1, 1, 2))),
                "mode": kw_copy.pop("mode", "zero-pad"),
                "output_size": kw_copy.pop("output_size", (256, 256)),
                **kw_copy,
            }
            new_images.append(transform(data)["image"])
        images = new_images
    try:
        return validate_images(torch.cat(images))
    except RuntimeError as e:
        raise RuntimeError(f"All images should have the same shape.") from e
