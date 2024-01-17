import unittest

import torch

from dgs.models.similarity.combined import DynamicallyGatedSimilarities


class TestCombined(unittest.TestCase):
    def test_DGS(self):
        for a_shape, s_shape in [
            ((), (7, 20)),
            ((1,), (7, 20)),
            ((1, 1), (7, 20)),
            ((1, 1, 1), (7, 20)),
            ((1, 7, 1), (7, 20)),
            ((7,), (7, 20)),
            ((7, 1), (7, 20)),
        ]:
            with self.subTest(msg=f"a_shape: {a_shape}, s_shape: {s_shape}"):
                self.assertTrue(
                    torch.allclose(
                        DynamicallyGatedSimilarities.forward(
                            torch.ones(a_shape) * 0.2, torch.ones(s_shape).float(), torch.ones(s_shape).float()
                        ),
                        torch.ones(s_shape),
                    )
                )

    def test_forward_exceptions(self):
        for a_shape, s1_shape, s2_shape, exception, err_str in [
            ((1,), (7, 21), (7, 20), ValueError, "s1 and s2 should have the same shapes, but are"),
            ((1,), (7, 20, 1), (7, 20), ValueError, "s1 and s2 should have the same shapes, but are"),
            ((1, 7), (7, 20), (7, 20), ValueError, "If alpha is two dimensional, the second dimension has to be 1"),
            ((1, 2, 1), (7, 20), (7, 20), ValueError, "alpha has the wrong shape"),
            ((8,), (7, 20), (7, 20), ValueError, "If the length of the first dimension of alpha is not 1, "),
            ((2, 3, 4), (7, 20), (7, 20), ValueError, "alpha has the wrong shape"),
        ]:
            with self.subTest(msg=f"a_shape: {a_shape}, s1_shape: {s2_shape}, s1_shape: {s2_shape}"):
                with self.assertRaises(exception):
                    DynamicallyGatedSimilarities.forward(
                        torch.ones(a_shape) * 0.2, torch.ones(s1_shape).float(), torch.ones(s2_shape).float()
                    )


if __name__ == "__main__":
    unittest.main()
