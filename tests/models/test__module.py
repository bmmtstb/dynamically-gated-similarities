import shutil
import unittest
from unittest.mock import patch

import torch
from easydict import EasyDict
from torch import nn

from dgs.models.module import BaseModule, module_validations as base_module_validation
from dgs.utils.config import fill_in_defaults
from dgs.utils.constants import PRECISION_MAP
from dgs.utils.exceptions import InvalidParameterException, ValidationException
from dgs.utils.files import mkdir_if_missing, to_abspath
from dgs.utils.types import Config, Device
from helper import get_test_config, test_multiple_devices

TEST_CFG: Config = EasyDict(
    {
        "name": "TestModel",
        "description": "Test Description",
        "print_prio": "DEBUG",
        "device": "cpu",
        "log_dir": "./tests/test_data/logs/",
        "gpus": "",
        "sp": True,
        "is_training": False,
        "num_workers": 0,
    }
)


def _def_repl(key: str, value: any) -> Config:
    """Create a copy of the test configuration and replace the given key with the new value

    Args:
        key: key to replace
        value: value to set

    Returns:
        A modified copy of `TEST_CFG`.
    """
    new_cfg: Config = EasyDict(TEST_CFG.copy())
    new_cfg[key] = value
    return new_cfg


class TestBaseModule(unittest.TestCase):
    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_validate_valid_config(self):
        for cfg in [
            TEST_CFG,
            _def_repl("device", "cpu"),
            _def_repl("device", "cuda"),
            _def_repl("print_prio", "INFO"),
            _def_repl("print_prio", "WARNING"),
            _def_repl("print_prio", "ERROR"),
            _def_repl("num_workers", 4),
            _def_repl("training", True),
            _def_repl("training", False),
            _def_repl("precision", "float64"),
        ]:
            with self.subTest(msg=str(cfg)):
                BaseModule(config=cfg, path=[]).validate_params(base_module_validation, "config")

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_validate_invalid_config(self):
        for cfg, err_str, msg in [
            (_def_repl("device", "gpu"), "'device' is not valid. Value is 'gpu'", "gpu invalid should be cuda"),
            (_def_repl("device", ""), "'device' is not valid", "empty device"),
            (_def_repl("device", None), "'device' is not valid", "None device"),
            (_def_repl("print_prio", ""), "'print_prio' is not valid", "empty print priority"),
            (_def_repl("print_prio", "None"), "'print_prio' is not valid", "caps not valid"),
            (_def_repl("print_prio", None), "'print_prio' is not valid", "None print priority"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(InvalidParameterException) as context:
                    BaseModule(config=cfg, path=[]).validate_params(base_module_validation, "config")
                self.assertTrue(err_str in str(context.exception))

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_validate_params(self):
        path = "dummy"
        for validations, data, valid in [
            ({"T": ["None"]}, {"T": None}, True),
            ({"T": ["optional"]}, {}, True),
            ({"T": [lambda _: True]}, {"T": None}, True),
            ({"T": [lambda _: False]}, {"T": None}, False),
            ({"T": [lambda x: x is False]}, {"T": False}, True),
            ({"T": ["number", ("gt", 0)]}, {}, False),
            ({"T": ["optional"]}, {"T": None}, True),
            ({"T": ["optional", "number", ("gt", 0)]}, {"T": 2}, True),
            ({"T": ["optional", "number", ("gt", 0)]}, {"T": "No Number!"}, False),
            ({"T1": ["optional"], "T2": ["optional"], "T3": ["optional"]}, {"T1": "No Number!", "T2": 2}, True),
        ]:
            with self.subTest(msg=f"validations: {validations}, data: {data}"):
                m = BaseModule(config=_def_repl(path, data), path=[path])
                try:
                    m.validate_params(validations)
                    self.assertTrue(valid)
                except InvalidParameterException:
                    self.assertFalse(valid)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_validate_params_raises(self):
        path = "dummy"
        for validations, data, exception in [
            ({"T": []}, {"T": None}, ValidationException),
            ({"T": [None]}, {"T": None}, ValidationException),
        ]:
            with self.subTest(msg=f"validations: {validations}, data: {data}, exception: {exception}"):
                m = BaseModule(config=_def_repl(path, data), path=[path])
                with self.assertRaises(exception):
                    m.validate_params(validations)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    @test_multiple_devices
    def test_configure_torch_model(self, device: Device):
        for module, train in [
            (nn.Linear(10, 2), True),
            (nn.Linear(10, 2), False),
        ]:
            with self.subTest(msg="module: {}, train: {}".format(module, train)):
                m = BaseModule(
                    config=fill_in_defaults(
                        {"name": "TestName", "is_training": train, "device": device},
                        default_cfg=get_test_config(),
                    ),
                    path=[],
                )
                self.assertEqual(module.bias.device.type, torch.device("cpu").type)
                m.configure_torch_module(module, train)
                self.assertEqual(module.training, train)
                self.assertEqual(module.bias.device.type, device.type)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_name_safe(self):
        for name, safe in [
            ("Dummy", "Dummy"),
            ("S p a c e s", "S-p-a-c-e-s"),
            ("D.o.t.t.e.d", "D_o_t_t_e_d"),
            ("U_n_d_e_r_s_c_o_r_e_d", "U_n_d_e_r_s_c_o_r_e_d"),
            ("T-i-l-e-d", "T-i-l-e-d"),
        ]:
            with self.subTest(msg="name: {}, safe: {}".format(name, safe)):
                cfg = fill_in_defaults(
                    {"name": name},
                    default_cfg=get_test_config(),
                )
                m = BaseModule(config=cfg, path=[])

                self.assertEqual(m.name, name)
                self.assertEqual(m.name_safe, safe)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_no_precision_in_cfg(self):
        cfg = get_test_config()
        m = BaseModule(config=cfg, path=[])
        self.assertEqual(m.precision, torch.float32)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_precision_raises(self):
        cfg = fill_in_defaults(
            {"precision": "dummy"},
            default_cfg=get_test_config(),
        )
        with self.assertRaises(InvalidParameterException) as e:
            _ = BaseModule(config=cfg, path=[])
        self.assertTrue("parameter 'precision' is not valid" in str(e.exception), msg=e.exception)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_precision(self):
        for precision, dtype in [
            (int, torch.int),
            (float, torch.float),
            (torch.float, torch.float),
            (torch.float32, torch.float32),
            (torch.float64, torch.float64),
        ]:
            with self.subTest(msg="precision: {}, dtype: {}".format(precision, dtype)):
                cfg = fill_in_defaults(
                    {"precision": precision},
                    default_cfg=get_test_config(),
                )
                m = BaseModule(config=cfg, path=[])
                self.assertEqual(m.precision, dtype)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_precision_constants(self):
        for precision, dtype in PRECISION_MAP.items():
            with self.subTest(msg="precision: {}, dtype: {}".format(precision, dtype)):
                cfg = fill_in_defaults(
                    {"precision": precision},
                    default_cfg=get_test_config(),
                )
                m = BaseModule(config=cfg, path=[])
                self.assertEqual(m.precision, dtype)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_terminate_module(self):
        m = BaseModule(config=get_test_config(), path=[])
        self.assertTrue(hasattr(m, "logger"))
        m.terminate()
        self.assertFalse(hasattr(m, "logger"))

    def setUp(self):
        mkdir_if_missing("./tests/test_data/TEST_logs/")

    def tearDown(self):
        shutil.rmtree(to_abspath("./tests/test_data/TEST_logs/"))


if __name__ == "__main__":
    unittest.main()
