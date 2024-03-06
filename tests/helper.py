"""
Helper methods for tests.
"""

import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO
from typing import Iterable

import torch

from dgs.default_config import cfg as default_config
from dgs.utils.image import load_image
from dgs.utils.types import Config, Device, Image
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


def load_test_image(filename: str) -> Image:
    """Given the filename of an image in tests/test_data folder, load, validate and return it."""
    return validate_images(load_image(os.path.join("./tests/test_data/", filename)))


def load_test_images(filenames: Iterable[str], force_reshape: bool = False, **kwargs) -> Image:
    """Given the filename of an image in tests/test_data folder, load, validate and return it."""
    return load_image(
        tuple(os.path.join("./tests/test_data/", fn) for fn in filenames), force_reshape=force_reshape, **kwargs
    )


@contextmanager
def capture_stdout(command, *args, **kwargs):
    """Context manager for checking print to stdout."""
    out, sys.stdout = sys.stdout, StringIO()
    try:
        command(*args, **kwargs)
        sys.stdout.seek(0)
        yield sys.stdout.read()
    finally:
        sys.stdout = out


def get_test_config() -> Config:
    """Get the default configuration for tests.
    Will replace a few values to keep all the data local in the tests folder.
    """
    cfg = deepcopy(default_config)

    cfg.name = "Test"
    cfg.log_dir = "./tests/test_data/logs/"

    return cfg
