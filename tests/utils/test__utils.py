import unittest

import numpy as np
import torch

from dgs.utils.types import Device
from dgs.utils.utils import torch_to_numpy
from helper import test_multiple_devices


class TestUtils(unittest.TestCase):
    @test_multiple_devices
    def test_torch_to_np(self, device: Device):
        for torch_tensor, numpy_array in [
            (torch.ones((5, 2), device=device, dtype=torch.int32), np.ones((5, 2), dtype=np.int32)),
        ]:
            with self.subTest(msg=f"torch_tensor: {torch_tensor}, numpy_array: {numpy_array}"):
                self.assertTrue(np.array_equal(torch_to_numpy(torch_tensor), numpy_array))


if __name__ == "__main__":
    unittest.main()
