"""
Helper methods for tests.
"""
import os

import torch

from dgs.utils.image import load_image
from dgs.utils.types import Device, TVImage
from dgs.utils.validation import validate_images


def test_multiple_devices(func: callable) -> callable:
    """
    Decorator to run given test on all devices.
    Check whether CUDA is available and if not use CPU only.

    Args:
        func: The decorated function

    Returns:
        Decorated function with additional device argument
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
