import unittest

import numpy as np
import torch

from dgs.models import BaseModule, get_combine_module
from dgs.models.combine.combine import CombineSimilaritiesModule, DynamicallyGatedSimilarities, StaticAlphaCombine
from dgs.utils.config import fill_in_defaults
from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.types import Device
from helper import get_test_config, test_multiple_devices


class TestDGS(unittest.TestCase):
    default_cfg = get_test_config()

    def test_dgs_modules_class(self):
        for name, mod_class, kwargs in [
            ("DGS", DynamicallyGatedSimilarities, {}),
            ("static_alpha", StaticAlphaCombine, {"alpha": [0.4, 0.3, 0.3]}),
        ]:
            with self.subTest(msg="name: {}, module: {}, kwargs: {}".format(name, mod_class, kwargs)):
                module = get_combine_module(name)
                self.assertEqual(module, mod_class)

                cfg = fill_in_defaults({"dgs": kwargs}, default_cfg=self.default_cfg)
                module = module(config=cfg, path=["dgs"])

                self.assertTrue(isinstance(module, CombineSimilaritiesModule))
                self.assertTrue(isinstance(module, BaseModule))

        with self.assertRaises(InvalidParameterException) as e:
            _ = get_combine_module("dummy")
        self.assertTrue("Unknown combine similarities module with name" in str(e.exception), msg=e.exception)

    def test_dgs_init(self):
        m = DynamicallyGatedSimilarities(config=self.default_cfg, path=["weighted_similarity"])
        self.assertTrue(isinstance(m, CombineSimilaritiesModule))
        self.assertTrue(isinstance(m, BaseModule))

    @test_multiple_devices
    def test_dgs_forward(self, device: Device):
        N = 7
        T = 21
        default_cfg = get_test_config()
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
                    config=fill_in_defaults({"device": device}, default_cfg), path=["weighted_similarity"]
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
                dgs = DynamicallyGatedSimilarities(config=self.default_cfg, path=["weighted_similarity"])
                with self.assertRaises(exception_type) as e:
                    dgs.forward(*sn, alpha=alpha)
                self.assertTrue(msg in str(e.exception), msg=e.exception)


class TestConstantAlpha(unittest.TestCase):
    default_cfg = get_test_config()

    def test_constant_alpha_init(self):
        for alpha in [[0.9, 0.1], [0.5, 0.5], [0.1 for _ in range(10)]]:
            with self.subTest(msg="alpha: {}".format(alpha)):
                m = StaticAlphaCombine(
                    config=fill_in_defaults({"weighted_similarity": {"alpha": alpha}}, self.default_cfg),
                    path=["weighted_similarity"],
                )
                self.assertTrue(isinstance(m, CombineSimilaritiesModule))

    def test_constant_alpha_init_exceptions(self):
        for alpha, exp, text in [
            ([1.0], InvalidParameterException, "parameter 'alpha' is not valid"),
            ([0.5, 0.4], InvalidParameterException, "parameter 'alpha' is not valid. Used a custom validation"),
            ([1 / 11 for _ in range(10)], InvalidParameterException, "parameter 'alpha' is not valid"),
        ]:
            with self.subTest(msg="alpha: {}".format(alpha)):
                cfg = fill_in_defaults({"sim": {"alpha": alpha}}, self.default_cfg)
                with self.assertRaises(exp) as e:
                    _ = StaticAlphaCombine(config=cfg, path=["sim"])
                self.assertTrue(text in str(e.exception), msg=e.exception)

    def test_constant_alpha_forward(self):
        N = 7
        T = 21

        for alpha, sn, result in [
            ([1.0, 0.0], (torch.ones((N, T)), torch.zeros((N, T))), torch.ones((N, T))),
            ([0.5, 0.5], (torch.ones((N, T)), torch.zeros((N, T))), 0.5 * torch.ones((N, T))),
            ([0.7, 0.3], (torch.ones((N, T)), -1 * torch.ones((N, T))), 0.4 * torch.ones((N, T))),
            (
                [0.2, 0.8],
                (torch.tensor([[5, 0], [0, 5]]).float(), torch.tensor([[0, 1.25], [1.25, 0]]).float()),
                torch.ones((2, 2)),
            ),
            (
                [0.25, 0.25, 0.25, 0.25],
                (torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T)), torch.ones((N, T))),
                torch.ones((N, T)),
            ),
            (
                [0.1, 0.2, 0.3, 0.4],
                (torch.ones((N, T)), torch.ones((N, T)), -1 * torch.ones((N, T)), torch.zeros((N, T))),
                torch.zeros((N, T)),
            ),
        ]:
            with self.subTest(msg="alpha: {}, sn: {}".format(alpha, sn)):
                m = StaticAlphaCombine(
                    config=fill_in_defaults({"weighted_similarity": {"alpha": alpha}}, self.default_cfg),
                    path=["weighted_similarity"],
                )
                self.assertTrue(torch.allclose(m.forward(*sn), result))

    def test_constant_alpha_forward_exceptions(self):
        N = 7
        T = 21

        for alpha, sn, exception, err_msg in [
            ([0.5, 0.5], (torch.ones((N, T)), np.ones((N, T))), TypeError, "All the values in args should be tensors"),
            ([0.5, 0.5], (torch.ones((N + 1, T)), torch.ones((N, T))), ValueError, "shapes of every tensor should"),
            ([0.5, 0.5], (torch.ones((N, T + 1)), torch.ones((N, T))), ValueError, "shapes of every tensor should"),
            ([0.5, 0.5], (torch.ones((N, T)), torch.ones((N + 1, T))), ValueError, "shapes of every tensor should"),
            ([0.5, 0.5], (torch.ones((N, T)), torch.ones((N, T + 1))), ValueError, "shapes of every tensor should"),
        ]:
            with self.subTest(msg="alpha: {}, sn: {}, exp: {}, err_msg: {}".format(alpha, sn, exception, err_msg)):
                with self.assertRaises(exception) as e:
                    m = StaticAlphaCombine(
                        config=fill_in_defaults({"weighted_similarity": {"alpha": alpha}}, self.default_cfg),
                        path=["weighted_similarity"],
                    )
                    m.forward(*sn)
                self.assertTrue(err_msg in str(e.exception))


if __name__ == "__main__":
    unittest.main()
