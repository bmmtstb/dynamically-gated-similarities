"""
helper methods for tests
"""
import torch

from dgs.utils.types import Device


def test_multiple_devices(func: callable) -> callable:
    """
    Decorator to run given test on all devices.
    Check whether CUDA is available and if not use CPU only.

    Args:
        func: The decorated function

    Returns:
        Decorated function with additional device argument
    """
    if torch.cuda.is_available():
        devices: list[Device] = ["cpu", "cuda"]
    else:
        devices: list[Device] = ["cpu"]

    def device_wrapper(cls, *args, **kwargs):
        """wrapper for decorator"""
        for device in devices:
            func(cls, device=device, *args, **kwargs)

    return device_wrapper
