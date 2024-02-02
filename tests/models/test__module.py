import unittest
from unittest.mock import patch

import torch
from easydict import EasyDict
from torch import nn

from dgs.models.module import BaseModule, module_validations as base_module_validation
from dgs.utils.config import fill_in_defaults
from dgs.utils.exceptions import InvalidParameterException, ValidationException
from dgs.utils.types import Config, Device
from helper import test_multiple_devices

TEST_CFG: Config = EasyDict(
    {
        "name": "TestModel",
        "print_prio": "debug",
        "device": "cpu",
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
            _def_repl("print_prio", "none"),
            _def_repl("print_prio", "normal"),
            _def_repl("print_prio", "all"),
            _def_repl("gpus", None),
            _def_repl("gpus", [0, 1, 2]),
            _def_repl("gpus", "0,1,2,3"),
            _def_repl("gpus", ""),
            _def_repl("num_workers", None),
            _def_repl("num_workers", 4),
            _def_repl("sp", True),
            _def_repl("sp", False),
            _def_repl("training", True),
            _def_repl("training", False),
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
            (_def_repl("gpus", ["0", "1", "2"]), "Used a custom validation", "gpus as list of str"),
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
    def test_print(self):
        for prio, module_prio, allowed in [
            ("normal", "none", False),
            ("debug", "none", False),
            ("all", "none", False),
            ("normal", "normal", True),
            ("debug", "normal", False),
            ("all", "normal", False),
            ("normal", "debug", True),
            ("debug", "debug", True),
            ("all", "debug", False),
            ("normal", "all", True),
            ("debug", "all", True),
            ("all", "all", True),
        ]:
            with self.subTest(msg=f"Printing prio: {prio}, Module prio {module_prio}"):
                self.assertEqual(
                    BaseModule(config=_def_repl("print_prio", module_prio), path=[]).can_print(prio), allowed
                )

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_print_errors(self):
        for prio, module_prio, err_str in [
            ("none", "none", "To print with priority of none doesn't make sense..."),
            ("none", "normal", "To print with priority of none doesn't make sense..."),
            ("none", "debug", "To print with priority of none doesn't make sense..."),
            ("none", "all", "To print with priority of none doesn't make sense..."),
            ("invalid", "all", "invalid is not in "),
        ]:
            with self.subTest(msg=f"Printing prio: {prio}, Module prio {module_prio}"):
                with self.assertRaises(ValueError) as context:
                    BaseModule(config=_def_repl("print_prio", module_prio), path=[]).can_print(prio)
                self.assertTrue(err_str in str(context.exception))

    @patch.multiple(BaseModule, __abstractmethods__=set())
    @test_multiple_devices
    def test_configure_torch_model(self, device: Device):
        for module, train in [
            (nn.Linear(10, 2), True),
            (nn.Linear(10, 2), False),
        ]:
            with self.subTest(msg="module: {}, train: {}".format(module, train)):
                m = BaseModule(
                    config=fill_in_defaults({"name": "TestName", "is_training": train, "device": device}), path=[]
                )
                self.assertEqual(module.bias.device.type, torch.device("cpu").type)
                m.configure_torch_module(module, train)
                self.assertEqual(module.training, train)
                self.assertEqual(module.bias.device.type, device.type)


if __name__ == "__main__":
    unittest.main()
