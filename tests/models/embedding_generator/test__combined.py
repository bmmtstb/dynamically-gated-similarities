import unittest

import numpy as np
import torch

from dgs.default_config import cfg as default_cfg
from dgs.models.similarity import DynamicallyGatedSimilarities, StaticAlphaWeightingModule
from dgs.utils.config import fill_in_defaults
from dgs.utils.types import Device
from helper import test_multiple_devices


class TestDGS(unittest.TestCase):
    def test_dgs_init(self):
        _ = DynamicallyGatedSimilarities(config=default_cfg, path=["weighted_similarity"])

    @test_multiple_devices
    def test_dgs_forward(self, device: Device):
        N = 7
        T = 21
        for alpha, s1, s2, result in [
            (torch.tensor([0.7]), torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T))),
            (torch.tensor([[0.3]]), torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T))),
            (torch.ones(N) * 0.3, torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T))),
            (torch.ones((N, 1)) * 0.3, torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T))),
            (torch.ones((1, 1, N, 1)) * 0.3, torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T))),
        ]:
            with self.subTest(
                msg="alpha: {}, s1: {}, s2: {}, result: {}, device: {}".format(alpha, s1, s2, result, device)
            ):
                dgs = DynamicallyGatedSimilarities(
                    config=fill_in_defaults({"device": device}), path=["weighted_similarity"]
                )
                # send matrices to the respective device
                self.assertTrue(
                    torch.allclose(
                        dgs.forward(s1.to(device=device), s2.to(device=device), alpha=alpha.to(device=device)),
                        result.to(device=device),
                    )
                )

    def test_dgs_forward_raises(self):
        N = 7
        T = 21
        for alpha, sn, exception_type, msg in [
            (
                torch.tensor([-0.1]),
                (torch.ones((N, T)), torch.ones((N, T))),
                ValueError,
                "alpha should lie in the range",
            ),
            (
                torch.tensor([1.1]),
                (torch.ones((N, T)), torch.ones((N, T))),
                ValueError,
                "alpha should lie in the range",
            ),
            (torch.ones((2, 2, 1)), (torch.ones((N, T)), torch.ones((N, T))), ValueError, "alpha has the wrong shape"),
            (torch.ones((2, 2)), (torch.ones((N, T)), torch.ones((N, T))), ValueError, "If alpha is two dimensional"),
            (torch.ones(1), (torch.ones((N + 1, T)), torch.ones((N, T))), ValueError, "s1 and s2 should have the same"),
            (torch.ones(1), (torch.ones((N, T + 1)), torch.ones((N, T))), ValueError, "s1 and s2 should have the same"),
            (torch.ones(1), (torch.ones((N, T)), torch.ones((N + 1, T))), ValueError, "s1 and s2 should have the same"),
            (torch.ones(1), (torch.ones((N, T)), torch.ones((N, T + 1))), ValueError, "s1 and s2 should have the same"),
            (torch.tensor([-0.1]), (np.ones((N, T)), torch.ones((N, T))), TypeError, "All matrices should be torch"),
            (
                torch.ones((N + 1, 1)).float(),
                (torch.ones((N, T)).float(), torch.ones((N, T)).float()),
                ValueError,
                "the first dimension has to equal",
            ),
            (
                torch.tensor([-0.1]),
                (torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T))),
                ValueError,
                "There should be exactly two matrices in the tensors argument",
            ),
        ]:
            with self.subTest(
                msg="alpha: {}, s1: {}, s2: {}, excp: {}, msg: {}".format(
                    alpha.shape, sn[0].shape, sn[0].shape, exception_type, msg
                )
            ):
                dgs = DynamicallyGatedSimilarities(config=default_cfg, path=["weighted_similarity"])
                with self.assertRaises(exception_type) as e:
                    dgs.forward(*sn, alpha=alpha)
                self.assertTrue(msg in str(e.exception), msg=e.exception)


class TestConstantAlpha(unittest.TestCase):
    def test_constant_alpha_init(self):
        for alpha in [[1], [0.5, 0.5], [1 / 10 for _ in range(10)]]:
            with self.subTest(msg="alpha: {}".format(alpha)):
                _ = StaticAlphaWeightingModule(
                    config=fill_in_defaults({"weighted_similarity": {"alpha": alpha}}), path=["weighted_similarity"]
                )

    def test_constant_alpha_init_exceptions(self):
        for alpha in [[1], [0.5, 0.5], [1 / 10 for _ in range(10)]]:
            with self.subTest(msg="alpha: {}".format(alpha)):
                _ = StaticAlphaWeightingModule(
                    config=fill_in_defaults({"weighted_similarity": {"alpha": alpha}}), path=["weighted_similarity"]
                )

    def test_constant_alpha_forward(self):
        N = 7
        T = 21

        for alpha, sn, result in [
            ([1], (torch.ones((N, T)),), torch.ones((N, T))),
            ([0.5, 0.5], (torch.ones((N, T)), torch.ones((N, T))), torch.ones((N, T))),
        ]:
            with self.subTest(msg="alpha: {}, sn: {}".format(alpha, sn)):
                m = StaticAlphaWeightingModule(
                    config=fill_in_defaults({"weighted_similarity": {"alpha": alpha}}), path=["weighted_similarity"]
                )
                self.assertTrue(torch.allclose(m.forward(*sn), result))

    def test_constant_alpha_forward_exceptions(self):
        N = 7
        T = 21

        for alpha, sn, exception, err_msg in [
            ([1], (torch.ones((N, T)), torch.ones((N, T))), ValueError, "length of the similarity matrices"),
            ([0.5, 0.5], (torch.ones((N, T)), np.ones((N, T))), TypeError, "All the values in args should be tensors"),
            ([0.4, 0.4], (torch.ones((N, T)), torch.ones((N, T))), ValueError, "alpha should sum to 1"),
            ([0.5, 0.5], (torch.ones((N + 1, T)), torch.ones((N, T))), ValueError, "shapes of every tensor should"),
            ([0.5, 0.5], (torch.ones((N, T + 1)), torch.ones((N, T))), ValueError, "shapes of every tensor should"),
            ([0.5, 0.5], (torch.ones((N, T)), torch.ones((N + 1, T))), ValueError, "shapes of every tensor should"),
            ([0.5, 0.5], (torch.ones((N, T)), torch.ones((N, T + 1))), ValueError, "shapes of every tensor should"),
        ]:
            with self.subTest(msg="alpha: {}, sn: {}, exp: {}, err_msg: {}".format(alpha, sn, exception, err_msg)):
                with self.assertRaises(exception) as e:
                    m = StaticAlphaWeightingModule(
                        config=fill_in_defaults({"weighted_similarity": {"alpha": alpha}}), path=["weighted_similarity"]
                    )
                    m.forward(*sn)
                self.assertTrue(err_msg in str(e.exception))


if __name__ == "__main__":
    unittest.main()
