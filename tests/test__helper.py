import unittest

from dgs.utils.types import Device
from tests.helper import test_multiple_devices


class TestTestHelpers(unittest.TestCase):
    device_id: int = 0
    devices: list[Device] = ["cpu", "cuda"]

    @test_multiple_devices
    def test_test_multiple_devices(self, device):
        self.assertEqual(device, self.devices[self.device_id])
        self.device_id += 1


if __name__ == "__main__":
    unittest.main()
