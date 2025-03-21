import unittest
from unittest.mock import patch

import numpy as np
import torch as t
from torch import nn

from dgs.models.combine import (
    COMBINE_MODULES,
    get_combine_module,
    register_combine_module,
)
from dgs.models.combine.combine import CombineSimilaritiesModule
from dgs.models.combine.dynamic import AlphaCombine, DynamicAlphaCombine
from dgs.models.combine.static import StaticAlphaCombine
from dgs.models.module import BaseModule
from dgs.utils.config import fill_in_defaults
from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.nn import fc_linear
from dgs.utils.state import State
from dgs.utils.types import Device
from helper import get_test_config, test_multiple_devices
from tests.utils.state import B, DUMMY_DATA, DUMMY_DATA_BATCH


class TestCombineSimilaritiesModule(unittest.TestCase):

    default_cfg = fill_in_defaults(
        {
            "default": {"module_name": "dynamic_alpha", "alpha_modules": ["box_fc1"]},
            "box_fc1": {
                "module_name": "FullyConnectedAlpha",
                "name": "bbox",
                "hidden_layers": [4, 1],
                "bias": True,
                "act_func": None,
            },
        },
        get_test_config(),
    )

    def test_dynamic_alpha_class(self):
        for name, mod_class, kwargs in [
            ("alpha_combine", AlphaCombine, {}),
            ("dynamic_alpha", DynamicAlphaCombine, {"alpha_modules": ["box_fc1"]}),
            ("dynamic_alpha", DynamicAlphaCombine, {"softmax": False, "alpha_modules": ["box_fc1"]}),
            ("static_alpha", StaticAlphaCombine, {"alpha": [0.4, 0.3, 0.3]}),
        ]:
            with self.subTest(msg="name: {}, module: {}, kwargs: {}".format(name, mod_class, kwargs)):
                module = get_combine_module(name)
                self.assertEqual(module, mod_class)

                kwargs["module_name"] = name
                cfg = fill_in_defaults({"dgs": kwargs}, default_cfg=self.default_cfg)
                module = module(config=cfg, path=["dgs"])

                self.assertTrue(isinstance(module, CombineSimilaritiesModule))
                self.assertTrue(isinstance(module, BaseModule))

        with self.assertRaises(KeyError) as e:
            _ = get_combine_module("dummy")
        self.assertTrue("Instance 'dummy' is not defined in" in str(e.exception), msg=e.exception)

    def test_register_combine(self):
        with patch.dict(COMBINE_MODULES):
            for name, func, exception in [
                ("dummy", StaticAlphaCombine, False),
                ("dummy", StaticAlphaCombine, KeyError),
                ("new_dummy", StaticAlphaCombine, False),
            ]:
                with self.subTest(msg="name: {}, func: {}, except: {}".format(name, func, exception)):
                    if exception is not False:
                        with self.assertRaises(exception):
                            register_combine_module(name, func)
                    else:
                        register_combine_module(name, func)
                        self.assertTrue("dummy" in COMBINE_MODULES)
        self.assertTrue("dummy" not in COMBINE_MODULES)
        self.assertTrue("new_dummy" not in COMBINE_MODULES)


class TestDynamicAlphaCombine(unittest.TestCase):
    S = 2
    D = 3
    T = 7
    sim_sizes = [4, 1]

    default_cfg = fill_in_defaults(
        {
            "def_dynamic_alpha": {
                "module_name": "dynamic_alpha",
                "softmax": False,
                "alpha_modules": ["box_fc1", "pose_coco_multi"],
            },
            "box_fc1": {
                "module_name": "FullyConnectedAlpha",
                "name": "bbox",
                "hidden_layers": [4, 1],
                "bias": True,
                "act_func": None,
            },
            "pose_coco_multi": {
                "module_name": "SequentialCombinedAlpha",
                "name": "keypoints",
                "paths": [
                    {"Conv1d": {"in_channels": 17, "out_channels": 15, "kernel_size": 2, "groups": 1, "bias": True}},
                    "Flatten",
                    ["pose_coco_multi", "fc"],
                ],
                "fc": {
                    "module_name": "FullyConnectedAlpha",
                    "hidden_layers": [15, 1],
                    "bias": False,
                },
            },
        },
        get_test_config(),
    )

    @classmethod
    def setUpClass(cls):
        # default models with alpha modules with different sizes
        cls.default_models = nn.ModuleList(
            [fc_linear(hidden_layers=[cls.sim_sizes[i], 1], bias=False) for i in range(cls.S)]
        )
        for i in range(cls.S):
            nn.init.constant_(cls.default_models[i][0].weight, 1.0 / cls.sim_sizes[i])

        # constant models - uses sim_sizes[0] for all hidden layers
        cls.constant_models = nn.ModuleList(
            [fc_linear(hidden_layers=[cls.sim_sizes[0], 1], bias=False) for _ in range(cls.S)]
        )
        for i in range(cls.S):
            nn.init.constant_(cls.constant_models[i][0].weight, 1.0 / cls.sim_sizes[0])

    def setUp(self):
        self.assertEqual(len(self.sim_sizes), self.S)
        self.assertEqual(len(self.default_models), self.S)
        self.assertEqual(len(self.constant_models), self.S)

    def test_dynamic_alpha_cfg_init(self):
        m = DynamicAlphaCombine(config=self.default_cfg, path=["def_dynamic_alpha"])
        self.assertTrue(isinstance(m, CombineSimilaritiesModule))
        self.assertTrue(isinstance(m, BaseModule))
        self.assertFalse(hasattr(m, "alpha_model"))
        m.alpha_model = self.default_models
        self.assertTrue(hasattr(m, "alpha_model"))
        self.assertEqual(len(m.alpha_model), self.S)

    @test_multiple_devices
    def test_dynamic_alpha_forward(self, device: Device):
        # DAC module with alpha models of different sizes
        module = DynamicAlphaCombine(
            config=fill_in_defaults({"device": device}, self.default_cfg), path=["def_dynamic_alpha"]
        )
        module.alpha_model = self.constant_models.to(device=device)
        self.assertEqual(len(module.alpha_model), self.S)

        for tensors, state, result_shape in [
            # B=1 with batch size not given
            (t.ones((self.S, 1, self.T)), State(**DUMMY_DATA), t.Size((1, self.T))),
            (tuple(t.ones((1, self.T)) for _ in range(self.S)), State(**DUMMY_DATA), t.Size((1, self.T))),
            (t.ones((self.S, B, self.T)), State(**DUMMY_DATA_BATCH), t.Size((B, self.T))),
            (tuple(t.ones((B, self.T)) for _ in range(self.S)), State(**DUMMY_DATA_BATCH), t.Size((B, self.T))),
        ]:
            with self.subTest(
                msg="device: {}, B: {}, tens_t: {}, res_shape: {}".format(device, state.B, type(tensors), result_shape)
            ):
                # send state and tensors to the respective device
                state.to(device=device)
                if isinstance(tensors, (list, tuple)):
                    tensors = [tensor.to(device=device) for tensor in tensors]
                else:
                    tensors = tensors.to(device=device)

                r = module.forward(*tensors, s=state)
                self.assertEqual(r.shape, result_shape)


class TestDynamicAlphaCombineExceptions(unittest.TestCase):
    S = 2
    D = 3
    T = 5
    hl_input_size = 1

    def setUp(self):
        self.config = fill_in_defaults(
            {
                "def_dynamic_alpha": {
                    "module_name": "dynamic_alpha",
                    "softmax": False,
                    "alpha_modules": ["box_fc1", "pose_coco_multi"],
                },
                "box_fc1": {
                    "module_name": "FullyConnectedAlpha",
                    "name": "bbox",
                    "hidden_layers": [4, 1],
                    "bias": True,
                    "act_func": None,
                },
                "pose_coco_multi": {
                    "module_name": "SequentialCombinedAlpha",
                    "name": "keypoints",
                    "paths": [
                        {
                            "Conv1d": {
                                "in_channels": 17,
                                "out_channels": 15,
                                "kernel_size": 2,
                                "groups": 1,
                                "bias": True,
                            }
                        },
                        "Flatten",
                        ["pose_coco_multi", "fc"],
                    ],
                    "fc": {
                        "module_name": "FullyConnectedAlpha",
                        "hidden_layers": [15, 1],
                        "bias": False,
                    },
                },
            },
            get_test_config(),
        )

        alpha_module = nn.ModuleList(
            [fc_linear(hidden_layers=[self.hl_input_size, 1], bias=False) for _ in range(self.S)]
        )

        self.model = DynamicAlphaCombine(config=self.config, path=["def_dynamic_alpha"])

        if not hasattr(self.model, "alpha_model"):
            self.model.alpha_model = alpha_module
        if len(self.model.alpha_model) == 0:
            self.model.alpha_model.extend(alpha_module)
        self.assertEqual(len(self.model.alpha_model), self.S)
        for i in range(self.S):
            nn.init.constant_(self.model.alpha_model[i][0].weight, 1.0)

        self.dummy_t = tuple(t.ones((self.D, self.T)) for _ in range(self.S))
        self.dummy_t_single = t.ones((self.S, self.D, self.T))
        self.dummy_state = State(**DUMMY_DATA)
        self.dummy_state_batched = State(**DUMMY_DATA_BATCH)

    def tearDown(self):
        if hasattr(self.model, "alpha_model"):
            del self.model.alpha_model
        del self.model
        del self.config

    def test_type_error(self):
        # tensors
        with self.assertRaises(TypeError) as e:
            # noinspection PyTypeChecker
            self.model.forward(*(1.0,), s=self.dummy_state)  # tensors should be tuple of t.Tensor
        self.assertIn("All similarity matrices should be (float) tensors", str(e.exception))
        with self.assertRaises(TypeError) as e:
            # noinspection PyTypeChecker
            self.model.forward(1.0, s=self.dummy_state)  # tensors should be tuple of t.Tensor
        self.assertIn("All similarity matrices should be (float) tensors", str(e.exception))

        # state
        with self.assertRaises(TypeError) as e:
            # noinspection PyTypeChecker
            self.model.forward(*self.dummy_t, s="dummy")
        self.assertTrue("s should be a State. Got:" in str(e.exception), msg=e.exception)

    def test_value_error_tensor_shapes(self):
        with self.assertRaises(ValueError) as e:
            self.model.forward(t.tensor([[1.0]]), t.tensor([[1.0, 2.0]]), s=self.dummy_state)  # Different shapes
        self.assertIn("All similarity matrices should have the same shape", str(e.exception))

    def test_runtime_error_tensor_device(self):
        if t.cuda.is_available():
            tensors = list(self.dummy_t)
            tensors[0] = tensors[0].to(device="cpu")
            tensors[1] = tensors[1].to(device="cuda")
            with self.assertRaises(RuntimeError) as e:
                self.model.forward(*tensors, s=self.dummy_state)  # Different devices
            self.assertIn("All tensors should be on the same device", str(e.exception))

    def test_value_error_on_nof_tensors(self):
        # Mismatch in number of tensors against number of alpha models
        with self.assertRaises(ValueError) as e:
            self.model.forward(*(t.ones((self.D, self.T)) for _ in range(self.S + 1)), s=self.dummy_state)
        self.assertIn(f"There should be as many alpha models {self.S} as tensors {self.S + 1}.", str(e.exception))

    def test_value_error_on_4d_tensors(self):
        with self.assertRaises(ValueError) as e:
            self.model.forward(*(t.ones((self.S, self.D, self.T)) for _ in range(self.S)), s=self.dummy_state)
        self.assertIn(f"Expected a 3D tensor [S x D x T], but got a tensor with shape", str(e.exception))

    def test_state_not_set(self):
        with self.assertRaises(ValueError) as e:
            self.model.forward(*self.dummy_t)
        self.assertTrue("The state should be given" in str(e.exception), msg=e.exception)

    def test_runtime_error_state_device(self):
        if t.cuda.is_available():
            # tensor based
            with self.assertRaises(RuntimeError) as e:
                self.model.forward(
                    *self.dummy_t_single.cpu(), s=self.dummy_state.to(device="cuda")
                )  # Different devices
            self.assertIn("s should be on the same device as tensors", str(e.exception))

    def test_value_error_on_batch_mismatch(self):
        # Mismatch in number of alpha inputs against number of alpha models
        with self.assertRaises(ValueError) as e:
            self.model.forward(*self.dummy_t, s=self.dummy_state_batched)
        self.assertIn(f"The states batch size ({B}) should equal D of tensors ({self.D}).", str(e.exception))


class TestConstantAlpha(unittest.TestCase):
    N = 2
    D = 3
    T = 5

    default_cfg = fill_in_defaults(
        {
            "def_comb": {"module_name": "static_alpha", "alpha": [0.5, 0.5], "softmax": False},
        },
        get_test_config(),
    )

    def test_constant_alpha_init(self):
        for alpha in [[0.9, 0.1], [0.5, 0.5], [0.1 for _ in range(10)]]:
            with self.subTest(msg="alpha: {}".format(alpha)):
                m = StaticAlphaCombine(
                    config=self.default_cfg,
                    path=["def_comb"],
                )
                self.assertTrue(isinstance(m, CombineSimilaritiesModule))

    def test_constant_alpha_init_exceptions(self):
        for alpha, exp, text in [
            ([], InvalidParameterException, "parameter 'alpha' is not valid"),
            ([0.5, 0.4], InvalidParameterException, "parameter 'alpha' is not valid. Used a custom validation"),
            ([1 / 11 for _ in range(10)], InvalidParameterException, "parameter 'alpha' is not valid"),
        ]:
            with self.subTest(msg="alpha: {}".format(alpha)):
                cfg = fill_in_defaults({"sim": {"alpha": alpha, "module_name": "constant_alpha"}}, get_test_config())
                with self.assertRaises(exp) as e:
                    _ = StaticAlphaCombine(config=cfg, path=["sim"])
                self.assertTrue(text in str(e.exception), msg=e.exception)

    def test_constant_alpha_forward(self):
        N = 7
        T = 21

        for softmax in [True, False]:
            for alpha, sn, result in [
                ([1.0, 0.0], (t.ones((N, T)), t.zeros((N, T))), t.ones((N, T))),
                ([1.0], (t.ones((N, T)),), t.ones((N, T))),
                ([0.5, 0.5], (t.ones((N, T)), t.zeros((N, T))), 0.5 * t.ones((N, T))),
                ([0.7, 0.3], (t.ones((N, T)), -1 * t.ones((N, T))), 0.4 * t.ones((N, T))),
                (
                    [0.2, 0.8],
                    (t.tensor([[5, 0], [0, 5]]).float(), t.tensor([[0, 1.25], [1.25, 0]]).float()),
                    t.ones((2, 2)),
                ),
                (
                    [0.25, 0.25, 0.25, 0.25],
                    (t.ones((N, T)), t.ones((N, T)), t.ones((N, T)), t.ones((N, T))),
                    t.ones((N, T)),
                ),
                (
                    [0.1, 0.2, 0.3, 0.4],
                    (t.ones((N, T)), t.ones((N, T)), -1 * t.ones((N, T)), t.zeros((N, T))),
                    t.zeros((N, T)),
                ),
            ]:
                with self.subTest(msg="alpha: {}, sn: {}, softmax; {}".format(alpha, sn, softmax)):
                    m = StaticAlphaCombine(
                        config=fill_in_defaults(
                            {"def_comb": {"module_name": "static_alpha", "alpha": alpha, "softmax": softmax}},
                            get_test_config(),
                        ),
                        path=["def_comb"],
                    )
                    if softmax:
                        result = t.nn.functional.softmax(result, dim=-1)
                    self.assertTrue(t.allclose(m.forward(*sn), result), f"r: {result.shape}, sn: {t.stack(sn).shape}")

    def test_constant_alpha_forward_single_tensor_input(self):
        inp = t.stack(([t.ones((self.D, self.T)), t.zeros((self.D, self.T))]))
        alpha = [0.5, 0.5]

        m = StaticAlphaCombine(
            config=self.default_cfg,
            path=["def_comb"],
        )
        r = m(inp)

        self.assertEqual(list(r.shape), [self.D, self.T])
        self.assertTrue(t.allclose(r, 0.5 * t.ones((self.D, self.T))))

    def test_constant_alpha_forward_single_tensor_with_wrong_shape(self):
        inp = t.ones((2, self.D, self.T))  # size 2
        alpha = [0.4, 0.4, 0.2]  # size 3

        m = StaticAlphaCombine(
            config=fill_in_defaults({"module_name": "static_alpha", "def_comb": {"alpha": alpha}}, self.default_cfg),
            path=["def_comb"],
        )
        with self.assertRaises(ValueError) as e:
            _ = m(inp)
        self.assertTrue("of the tensors 2 should equal the length of alpha 3" in str(e.exception), msg=e.exception)

    def test_constant_alpha_forward_exceptions(self):

        for alpha, sn, exception, err_msg in [
            (
                [0.5, 0.5],
                (t.ones((self.D, self.T)), np.ones((self.D, self.T))),
                TypeError,
                "All the values in args should be tensors",
            ),
            (
                [0.5, 0.5],
                (t.ones((self.D + 1, self.T)), t.ones((self.D, self.T))),
                ValueError,
                "shapes of every tensor should",
            ),
            (
                [0.5, 0.5],
                (t.ones((self.D, self.T + 1)), t.ones((self.D, self.T))),
                ValueError,
                "shapes of every tensor should",
            ),
            (
                [0.5, 0.5],
                (t.ones((self.D, self.T)), t.ones((self.D + 1, self.T))),
                ValueError,
                "shapes of every tensor should",
            ),
            (
                [0.5, 0.5],
                (t.ones((self.D, self.T)), t.ones((self.D, self.T + 1))),
                ValueError,
                "shapes of every tensor should",
            ),
        ]:
            with self.subTest(msg="alpha: {}, sn: {}, exp: {}, err_msg: {}".format(alpha, sn, exception, err_msg)):
                with self.assertRaises(exception) as e:
                    m = StaticAlphaCombine(
                        config=fill_in_defaults({"def_comb": {"alpha": alpha}}, self.default_cfg),
                        path=["def_comb"],
                    )
                    m.forward(*sn)
                self.assertTrue(err_msg in str(e.exception))


class TestAlphaCombineExceptions(unittest.TestCase):
    N = 2
    D = 3
    T = 5

    config = fill_in_defaults({"alpha_comb": {"module_name": "alpha_combine", "softmax": False}}, get_test_config())
    model = AlphaCombine(config=config, path=["alpha_comb"])

    def setUp(self):
        # tensors values
        self.dummy_t = tuple(t.ones((self.D, self.T)) for _ in range(self.N))
        self.dummy_t_single = t.ones((self.N, self.D, self.T))
        # alpha values
        self.dummy_a = t.ones(self.N)
        self.dummy_a_list = [t.ones(1) for _ in range(self.N)]

    def test_type_error_for_tensors(self):
        with self.assertRaises(TypeError) as e:
            # noinspection PyTypeChecker
            _ = self.model.forward(*(1.0,), alpha=self.dummy_a)
        self.assertTrue("All similarity matrices should be (float) tensors" in str(e.exception), msg=e.exception)

    def test_device_mismatch_for_tensors(self):
        if t.cuda.is_available():
            tensors = list(self.dummy_t)
            tensors[0] = tensors[0].to(device="cpu")
            tensors[1] = tensors[1].to(device="cuda")
            with self.assertRaises(RuntimeError) as e:
                _ = self.model.forward(*tensors, alpha=self.dummy_a)
            self.assertTrue("All tensors should be on the same device" in str(e.exception), msg=e.exception)

    def test_type_error_for_alpha(self):
        with self.assertRaises(TypeError) as e:
            # noinspection PyTypeChecker
            _ = self.model.forward(*self.dummy_t, alpha=[0.5, 0.5])
        self.assertTrue("alpha should be a (float) tensor" in str(e.exception), msg=e.exception)

    def test_compared_lengths(self):
        with self.assertRaises(ValueError) as e:
            _ = self.model.forward(*self.dummy_t, alpha=t.ones(self.N + 1))
        self.assertTrue("should have the same length as the tensors" in str(e.exception), msg=e.exception)

    def test_compared_devices(self):
        if t.cuda.is_available():
            alpha = t.ones(self.N).to(device="cuda")
            with self.assertRaises(RuntimeError) as e:
                _ = self.model.forward(*self.dummy_t, alpha=alpha)
            self.assertTrue("alpha should be on the same device as the tensors" in str(e.exception), msg=e.exception)

    def test_alpha_not_set(self):
        with self.assertRaises(ValueError) as e:
            self.model.forward(*self.dummy_t)
        self.assertTrue("Alpha should be given" in str(e.exception), msg=e.exception)

    def test_value_error_on_2d_alpha(self):
        alpha = t.ones((self.N, self.D + 1))
        with self.assertRaises(ValueError) as e:
            _ = self.model.forward(*self.dummy_t, alpha=alpha)
        self.assertTrue("alpha should have shape [N x D], but got" in str(e.exception), msg=e.exception)

    def test_value_error_on_3d_alpha(self):
        alpha_1 = t.ones((self.N, self.D + 1, self.T))
        with self.assertRaises(ValueError) as e:
            _ = self.model.forward(*self.dummy_t, alpha=alpha_1)
        self.assertTrue("alpha should have shape [N x D x T], but got" in str(e.exception), msg=e.exception)

        alpha_2 = t.ones((self.N, self.D, self.T + 1))
        with self.assertRaises(ValueError) as e:
            _ = self.model.forward(*self.dummy_t, alpha=alpha_2)
        self.assertTrue("alpha should have shape [N x D x T], but got" in str(e.exception), msg=e.exception)

        alpha_3 = t.ones((self.N, self.D + 1, self.T + 1))
        with self.assertRaises(ValueError) as e:
            _ = self.model.forward(*self.dummy_t, alpha=alpha_3)
        self.assertTrue("alpha should have shape [N x D x T], but got" in str(e.exception), msg=e.exception)

    def test_not_implemented_error(self):
        with self.assertRaises(NotImplementedError) as e:
            self.model.forward(*self.dummy_t, alpha=t.ones((self.N, self.D)))
        self.assertIn("Alpha with shape [N x D] or [N x D x T] is not yet implemented", str(e.exception))

    def test_alpha_combine_cfg_init(self):
        m = AlphaCombine(config=self.config, path=["alpha_comb"])
        self.assertTrue(isinstance(m, AlphaCombine))
        self.assertTrue(isinstance(m, BaseModule))
        self.assertTrue(hasattr(m, "softmax"))

    @test_multiple_devices
    def test_dynamic_alpha_forward(self, device: Device):
        # DAC module with alpha models of different sizes
        model = AlphaCombine(config=fill_in_defaults({"device": device}, self.config), path=["alpha_comb"])

        for alpha, s_i, result in [
            (t.ones(self.N), t.ones((self.N, self.D, self.T)), t.ones((self.D, self.T)) * self.N),
            (t.ones(self.N), [t.ones((self.D, self.T)) for _ in range(self.N)], t.ones((self.D, self.T)) * self.N),
        ]:
            with self.subTest(
                msg="device: {}, type_ai: {}, type_si: {}, result: {}".format(
                    device, type(alpha), type(s_i), result.shape
                )
            ):
                # send matrices to the respective device
                alpha = alpha.to(device=device)
                if isinstance(s_i, (list, tuple)):
                    s_i = [s.to(device=device) for s in s_i]
                else:
                    s_i = s_i.to(device=device)
                result = result.to(device=device)

                self.assertEqual(t.Size([self.D, self.T]), result.shape)
                self.assertTrue(t.allclose(model.forward(*s_i, alpha=alpha), result))


if __name__ == "__main__":
    unittest.main()
