import unittest
from copy import deepcopy

from easydict import EasyDict

from dgs.default_config import cfg as default_config
from dgs.utils.config import fill_in_defaults, get_sub_config
from dgs.utils.exceptions import InvalidConfigException

EMPTY_CONFIG = EasyDict({})
SIMPLE_CONFIG = EasyDict({"foo": "foo foo", "second": "value"})

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
    def test_sub_config_still_correct_type(self):
        for cfg, path, new_type in [
            (NESTED_CONFIG, ["bar"], EasyDict),
            (NESTED_CONFIG, [], EasyDict),
            ({"bar": {"x": 1}}, ["bar", "x"], int),
            ({"bar": {"x": 1}}, ["bar"], dict),
            ({"bar": {"x": 1}}, [], dict),
        ]:
            with self.subTest(msg=f"Path: {path}, new_type: {new_type}, cfg: {cfg}"):
                self.assertIsInstance(get_sub_config(cfg, path), new_type)

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


class TestFillInConfig(unittest.TestCase):
    def test_fill_in_default(self):
        for curr_cfg, default_cfg, result, msg in [
            (EMPTY_CONFIG, SIMPLE_CONFIG, SIMPLE_CONFIG, "Adding to empty"),
            (NESTED_CONFIG, NESTED_CONFIG, NESTED_CONFIG, "Same will stay - nested"),
            (EMPTY_CONFIG, None, default_config, "get replaced by default config"),
            (EasyDict({"bar": {"x": 1}}), NESTED_CONFIG, NESTED_CONFIG, "on same value replace nested recursively"),
            ({"a": {"new": 1}}, {"a": 1, "b": 2}, {"a": {"new": 1}, "b": 2}, "Overwrite value with nested"),
            ({"a": 1, "b": 2}, {"a": {"old": 1}}, {"a": 1, "b": 2}, "Overwrite nested with value"),
            (EasyDict({"i": 1, "j": 2}), {"i": 0}, EasyDict({"i": 1, "j": 2}), "ED and D -> ED, duplicate"),
            ({"i": 1, "j": 2}, EasyDict({"i": 0, "k": 3}), EasyDict({"i": 1, "j": 2, "k": 3}), "D and ED -> ED, add"),
            ({"i": 1, "j": 2}, {"k": 3}, {"i": 1, "j": 2, "k": 3}, "D and D -> D, no overlap"),
            (
                EasyDict({"i": 1, "j": 2, "k": EasyDict({"a": 0})}),
                EasyDict({"k": 0, "i": EasyDict({"b": 0})}),
                EasyDict({"i": 1, "j": 2, "k": EasyDict({"a": 0})}),
                "ED and ED -> ED, nested values",
            ),
            (
                SIMPLE_CONFIG,
                NESTED_CONFIG,
                deepcopy(NESTED_CONFIG) | {"foo": "foo foo", "second": "value"},
                "Add nested to simple",
            ),
            (NESTED_CONFIG, SIMPLE_CONFIG, deepcopy(NESTED_CONFIG) | {"second": "value"}, "Add simple to nested"),
        ]:
            with self.subTest(msg=f"{msg}"):
                self.assertEqual(fill_in_defaults(curr_cfg, default_cfg), result)
        # test that no values were inserted into the defaults
        self.assertNotIn("second", NESTED_CONFIG)
        self.assertNotIn("bar", SIMPLE_CONFIG)
        self.assertNotIn("dog", SIMPLE_CONFIG)

    def test_fill_in_default_nested_update_value(self):
        current = EasyDict({"bar": {"deeper": {"pickaxe": {"iron": 100}}}})
        default = deepcopy(NESTED_CONFIG)
        result = deepcopy(NESTED_CONFIG)
        result["bar"]["deeper"]["pickaxe"]["iron"] = 100

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertEqual(NESTED_CONFIG["bar"]["deeper"]["pickaxe"]["iron"], 10)

    def test_fill_in_default_overwrite_dict(self):
        current = EasyDict({"bar": None})
        default = deepcopy(NESTED_CONFIG)
        result = deepcopy(NESTED_CONFIG)
        result["bar"] = None

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertEqual(NESTED_CONFIG["bar"]["deeper"]["pickaxe"]["iron"], 10)

    def test_fill_in_default_additional_value_in_current(self):
        current = EasyDict({"bar": {"deeper": {"pickaxe": {"copper": 100}}}})
        default = deepcopy(NESTED_CONFIG)
        result = deepcopy(NESTED_CONFIG)
        result["bar"]["deeper"]["pickaxe"]["copper"] = 100

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertNotIn("copper", NESTED_CONFIG["bar"]["deeper"]["pickaxe"])

    def test_fill_in_default_additional_value_in_default(self):
        current = deepcopy(NESTED_CONFIG)
        default = EasyDict({"bar": {"deeper": {"pickaxe": {"copper": 100}}}})
        result = deepcopy(NESTED_CONFIG)
        result["bar"]["deeper"]["pickaxe"]["copper"] = 100

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertNotIn("copper", NESTED_CONFIG["bar"]["deeper"]["pickaxe"])

    def test_fill_in_defaults_raises(self):
        with self.assertRaises(InvalidConfigException):
            # noinspection PyTypeChecker
            fill_in_defaults(None, SIMPLE_CONFIG)


if __name__ == "__main__":
    unittest.main()
