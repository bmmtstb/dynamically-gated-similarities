"""
helper methods for tests
"""

from dgs.utils.types import Device


def test_multiple_devices(func: callable) -> callable:
    """
    Decorator to run given test on all devices

    Args:
        func: the decorated function

    Returns:
        Decorated function with additional device argument
    """
    devices: list[Device] = ["cpu", "cuda"]

    def device_wrapper(cls, *args, **kwargs):
        """wrapper for decorator"""
        for device in devices:
            func(cls, device=device, *args, **kwargs)

    return device_wrapper
