import unittest

import torch
from torchvision import tv_tensors

from dgs.utils.visualization import torch_to_matplotlib


class TestVisualization(unittest.TestCase):
    def test_torch_to_matplotlib(self):
        B, C, H, W = 8, 3, 64, 64
        for tensor, out_shape in [
            (torch.ones(B, C, H, W), [B, H, W, C]),
            (tv_tensors.Image(torch.ones(B, C, H, W)), [B, H, W, C]),
            (torch.ones(C, H, W), [H, W, C]),
        ]:
            with self.subTest(msg="tensor: {}, out_shape: {}".format(tensor, out_shape)):
                m = torch_to_matplotlib(tensor)
                self.assertEqual(list(m.shape), list(out_shape))


if __name__ == "__main__":
    unittest.main()
