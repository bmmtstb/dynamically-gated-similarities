import unittest

import torch
from easydict import EasyDict

from dgs.utils.config import get_sub_config, validate_value

EMPTY_CONFIG = EasyDict({})
SIMPLE_CONFIG = EasyDict({"foo": "foo foo"})

NESTED_CONFIG = EasyDict(
    {
        "foo": 3,
        "bar": {
            "x": 1,
            "y": 2,
            "deeper": {
                "ore": "iron",
                "pickaxe": {"iron": 10, "diamond": 100.0},
            },
        },
        "dog": SIMPLE_CONFIG,
    }
)


class TestGetSubConfig(unittest.TestCase):
    def test_sub_config_still_easydict(self):
        self.assertIsInstance(NESTED_CONFIG["bar"], EasyDict)

    def test_empty_path(self):
        for in_config, path in [
            (EMPTY_CONFIG, []),
            (SIMPLE_CONFIG, []),
            (NESTED_CONFIG, []),
            (NESTED_CONFIG, None),
        ]:
            with self.subTest(msg=f"Path: {path}"):
                self.assertEqual(get_sub_config(in_config, path), in_config)

    def test_one_deep(self):
        for in_config, path, out_config in [
            (SIMPLE_CONFIG, ["foo"], "foo foo"),
            (NESTED_CONFIG, ["foo"], 3),
            (NESTED_CONFIG, ["dog"], SIMPLE_CONFIG),
        ]:
            with self.subTest(msg=f"Path: {path}"):
                self.assertEqual(get_sub_config(in_config, path), out_config)

    def test_recursive(self):
        for in_config, path, out_config in [
            (NESTED_CONFIG, ["bar", "x"], 1),
            (NESTED_CONFIG, ["bar", "y"], 2),
            (NESTED_CONFIG, ["bar", "deeper", "ore"], "iron"),
            (NESTED_CONFIG, ["bar", "deeper", "pickaxe", "iron"], 10),
            (NESTED_CONFIG, ["bar", "deeper", "pickaxe", "diamond"], float(100)),
            (NESTED_CONFIG, ["dog", "foo"], "foo foo"),
        ]:
            with self.subTest(msg=f"Path: {path}"):
                self.assertEqual(get_sub_config(in_config, path), out_config)

    def test_exceptions(self):
        for config, path in [
            (EMPTY_CONFIG, ["foo"]),
            (SIMPLE_CONFIG, ["bar"]),
            (SIMPLE_CONFIG, ["foo", "bar"]),
            (NESTED_CONFIG, ["foo", "bar"]),
            (NESTED_CONFIG, ["bar", "foo"]),
        ]:
            with self.subTest(msg=f"Path: {path}"):
                with self.assertRaises(KeyError):
                    get_sub_config(config, path)


class TestValidate(unittest.TestCase):
    def test_validate_value(self):
        for value, data, validation, result in [
            (None, ..., "None", True),
            (None, ..., "not None", False),
            (1, ..., "None", False),
            (1, ..., "not None", True),
            (1, (1, 2, 3), "in", True),
            (1.5, (1, 2, 3), "in", False),
            (1, [1, 2, 3], "in", True),
            ("1", [1, 2, 3], "in", False),
            (1, ["1", "2", "3"], "in", False),
            ("1", ["1", "2", "3"], "in", True),
            (1, ..., "float", False),
            (1.0, ..., "float", True),
            (1, float, "instance", False),
            (1.0, float, "instance", True),
            (["1", "2", "3"], 1, "contains", False),
            (["1", "2", "3"], "1", "contains", True),
        ]:
            with self.subTest(msg=f"value: {value}, data: {data}, validation: {validation}"):
                self.assertEqual(validate_value(value, data, validation), result)

    def test_nested_validations(self):
        for value, data, validation, valid in [
            ("cuda", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", True),
            ("cpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", True),
            ("gpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", False),
            (torch.device("cuda"), (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", True),
            ("cuda", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", True),
            ("cpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", True),
            ("gpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", False),
            (torch.device("cuda"), (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", True),
            (None, (("None", ...), ("not", ("not None", ...))), "and", True),
            (1, (("gte", 1), ("not None", ...), ("lte", 1.1), ("eq", 1), ("int", ...)), "and", True),
        ]:
            with self.subTest(msg=f"value {value}, validation {validation}"):
                self.assertEqual(validate_value(value, data, validation), valid)


if __name__ == "__main__":
    unittest.main()
