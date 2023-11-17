import unittest
from unittest.mock import patch

from easydict import EasyDict

from dgs.models.module import BaseModule, module_validations as base_module_validation
from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.types import Config

TEST_CFG: Config = EasyDict(
    {
        "device": "cpu",
        "print_prio": "debug",
    }
)


def _def_repl(key: str, value: any) -> Config:
    """
    Create a copy of the default configuration and replace the given key with the new value

    Args:
        key: key to replace
        value: value to set

    Returns:
        Modified copy of DEFAULT_CONFIG
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
        ]:
            with self.subTest(msg=str(cfg)):
                BaseModule(config=cfg, path=[]).validate_params(base_module_validation, "config")

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_validate_invalid_config(self):
        for cfg, err_str, msg in [
            (_def_repl("device", "gpu"), "device is not valid, was gpu", "gpu invalid, should be cuda"),
            (_def_repl("device", ""), "device is not valid", "empty device"),
            (_def_repl("device", None), "device is not valid", "None device"),
            (_def_repl("print_prio", ""), "print_prio is not valid", "empty print priority"),
            (_def_repl("print_prio", "None"), "print_prio is not valid", "caps not valid"),
            (_def_repl("print_prio", None), "print_prio is not valid", "None print priority"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(InvalidParameterException) as context:
                    BaseModule(config=cfg, path=[]).validate_params(base_module_validation, "config")
                self.assertTrue(err_str in str(context.exception))

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
                self.assertEqual(BaseModule(config=_def_repl("print_prio", module_prio), path=[]).print(prio), allowed)

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
                    BaseModule(config=_def_repl("print_prio", module_prio), path=[]).print(prio)
                self.assertTrue(err_str in str(context.exception))


if __name__ == "__main__":
    unittest.main()
