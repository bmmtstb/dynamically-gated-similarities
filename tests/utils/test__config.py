import os
import shutil
import unittest
from copy import deepcopy
from unittest.mock import patch

from dgs.models.module import BaseModule
from dgs.utils.config import (
    DEF_VAL,
    fill_in_defaults,
    get_sub_config,
    insert_into_config,
    load_config,
)
from dgs.utils.exceptions import InvalidConfigException, InvalidPathException
from dgs.utils.types import Config
from helper import get_test_config

EMPTY_CONFIG: Config = {}
SIMPLE_CONFIG: Config = {"foo": "foo foo", "second": "value"}
SUB_DICT: Config = {
    "x": 1,
    "y": 2,
    "deeper": {
        "ore": "iron",
        "pickaxe": {"iron": 10, "diamond": 100.0},
    },
}
NESTED_CONFIG: Config = {
    "foo": 3,
    "bar": SUB_DICT,
    "dog": SIMPLE_CONFIG,
}
TEST_CONFIG: Config = get_test_config()


class TestGetSubConfig(unittest.TestCase):
    def test_default_config_yaml(self):
        self.assertTrue(isinstance(DEF_VAL, dict))
        self.assertTrue(len(DEF_VAL.keys()) > 1)

    def test_sub_config_still_correct_type(self):
        for cfg, path, new_type in [
            (NESTED_CONFIG, ["bar"], dict),
            (NESTED_CONFIG, [], dict),
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
        for copy in [True, False]:
            for curr_cfg, default_cfg, result, msg, *_stays in [
                (deepcopy(EMPTY_CONFIG), deepcopy(SIMPLE_CONFIG), deepcopy(SIMPLE_CONFIG), "Add empty to simple", True),
                (deepcopy(SIMPLE_CONFIG), deepcopy(EMPTY_CONFIG), deepcopy(SIMPLE_CONFIG), "Add simple to empty"),
                (deepcopy(NESTED_CONFIG), deepcopy(NESTED_CONFIG), deepcopy(NESTED_CONFIG), "Same stay - nested", True),
                (
                    deepcopy(EMPTY_CONFIG),
                    deepcopy(TEST_CONFIG),
                    deepcopy(TEST_CONFIG),
                    "get replaced by default config",
                    True,
                ),
                (deepcopy(EMPTY_CONFIG), None, deepcopy(EMPTY_CONFIG), "get replaced by None", True),
                (
                    {"bar": {"x": 1}},
                    deepcopy(NESTED_CONFIG),
                    deepcopy(NESTED_CONFIG),
                    "on same value replace nested recursively",
                    True,
                ),
                ({"a": {"new": 1}}, {"a": 1, "b": 2}, {"a": {"new": 1}, "b": 2}, "Overwrite value with nested"),
                ({"a": 1, "b": 2}, {"a": {"old": 1}}, {"a": 1, "b": 2}, "Overwrite nested with value"),
                ({"i": 1, "j": 2}, {"i": 0}, {"i": 1, "j": 2}, "D and D -> D, duplicate"),
                ({"i": 1, "j": 2}, {"i": 0, "k": 3}, {"i": 1, "j": 2, "k": 3}, "D and D -> D, add"),
                ({"i": 1, "j": 2}, {"k": 3}, {"i": 1, "j": 2, "k": 3}, "D and D -> D, no overlap"),
                (
                    {"i": 1, "j": 2, "k": {"a": 0}},
                    {"k": 0, "i": {"b": 0}},
                    {"i": 1, "j": 2, "k": {"a": 0}},
                    "D and D -> D, nested values",
                ),
                (
                    deepcopy(SIMPLE_CONFIG),
                    deepcopy(NESTED_CONFIG),
                    deepcopy(NESTED_CONFIG) | {"foo": "foo foo", "second": "value"},
                    "Add nested to simple",
                ),
                (
                    deepcopy(NESTED_CONFIG),
                    deepcopy(SIMPLE_CONFIG),
                    deepcopy(NESTED_CONFIG) | {"second": "value"},
                    "Add simple to nested",
                ),
            ]:
                with self.subTest(msg=f"{msg}, copy: {copy}"):
                    orig = deepcopy(default_cfg)
                    new_cfg = fill_in_defaults(curr_cfg, default_cfg, copy=copy)
                    self.assertEqual(new_cfg, result, f"new: {new_cfg}\nexpected: {result}")
                    # orig should stay the same
                    # and catch the special cases:
                    # - adding an empty cfg
                    # - curr and new are the same
                    if copy or (len(_stays) and _stays[0]):
                        self.assertEqual(orig, default_cfg)
                    else:
                        self.assertNotEqual(orig, default_cfg)
        # test that no values were inserted into the defaults
        self.assertNotIn("second", NESTED_CONFIG)
        self.assertNotIn("bar", SIMPLE_CONFIG)
        self.assertNotIn("dog", SIMPLE_CONFIG)
        for value in ["second", "bar", "dog"]:
            self.assertNotIn(value, TEST_CONFIG)

    def test_fill_in_default_nested_update_value(self):
        current = {"bar": {"deeper": {"pickaxe": {"iron": 100}}}}
        default = deepcopy(NESTED_CONFIG)
        result = deepcopy(NESTED_CONFIG)
        result["bar"]["deeper"]["pickaxe"]["iron"] = 100

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertEqual(NESTED_CONFIG["bar"]["deeper"]["pickaxe"]["iron"], 10)

    def test_fill_in_default_overwrite_dict(self):
        current = {"bar": None}
        default = deepcopy(NESTED_CONFIG)
        result = deepcopy(NESTED_CONFIG)
        result["bar"] = None

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertEqual(NESTED_CONFIG["bar"]["deeper"]["pickaxe"]["iron"], 10)

    def test_fill_in_default_additional_value_in_current(self):
        current = {"bar": {"deeper": {"pickaxe": {"copper": 100}}}}
        default = deepcopy(NESTED_CONFIG)
        result = deepcopy(NESTED_CONFIG)
        result["bar"]["deeper"]["pickaxe"]["copper"] = 100

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertNotIn("copper", NESTED_CONFIG["bar"]["deeper"]["pickaxe"])

    def test_fill_in_default_additional_value_in_default(self):
        current = deepcopy(NESTED_CONFIG)
        default = {"bar": {"deeper": {"pickaxe": {"copper": 100}}}}
        result = deepcopy(NESTED_CONFIG)
        result["bar"]["deeper"]["pickaxe"]["copper"] = 100

        self.assertEqual(fill_in_defaults(current, default), result)
        self.assertNotIn("copper", NESTED_CONFIG["bar"]["deeper"]["pickaxe"])

    def test_fill_in_defaults_raises(self):
        with self.assertRaises(InvalidConfigException):
            # noinspection PyTypeChecker
            fill_in_defaults(None, SIMPLE_CONFIG)


class TestLoadConfig(unittest.TestCase):
    def test_load_config_dict(self):
        cfg = load_config("./tests/test_data/configs/test_config.yaml")
        self.assertIsInstance(cfg, dict)
        self.assertEqual(cfg["device"], "cpu")
        self.assertTrue(cfg["dummy_config"]["kwargs"]["more_data"], "Nesting did not get saved correctly.")
        self.assertListEqual(cfg["dummy_config"]["kwargs"]["even_more_data"], [1, 2, 3, 4])
        self.assertTupleEqual(cfg["dummy_config"]["kwargs"]["tuple_data"], (1, 2, 3))
        self.assertEqual(cfg["dummy_config"]["kwargs"]["none_data"], None)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_load_all_yaml_in_configs_dir(self):
        default_cfg = get_test_config()
        abs_path = "./configs/"
        paths = [
            os.path.normpath(os.path.join(abs_path, child_path))
            for child_path in os.listdir(abs_path)
            if child_path.endswith(".yaml") or child_path.endswith(".yml")
        ]
        for path in paths:
            with self.subTest(msg=f"path: {path}"):
                # make sure to change log_dir, so tests do not create empty log folders
                cfg = fill_in_defaults(
                    {"log_dir": f"./tests/test_data/TEST_loader/{path.split('.')[0]}/"},
                    load_config(path),
                )

                self.assertIsInstance(cfg, dict)
                self.assertIn("name", cfg)
                name = cfg["name"]
                cfg_w_def = fill_in_defaults(cfg, default_cfg=default_cfg)
                b = BaseModule(cfg_w_def, [])
                self.assertIsInstance(b, BaseModule)
                self.assertEqual(b.name, name)
        shutil.rmtree("./tests/test_data/TEST_loader/", ignore_errors=True)

    @patch.multiple(BaseModule, __abstractmethods__=set())
    def test_load_all_yaml_in_test_data(self):
        default_cfg = get_test_config()
        abs_path = "./tests/test_data/configs/"
        paths = [
            os.path.normpath(os.path.join(abs_path, child_path))
            for child_path in os.listdir(abs_path)
            if child_path.endswith(".yaml") or child_path.endswith(".yml")
        ]
        self.assertTrue(len(paths) > 2)
        for path in paths:
            with self.subTest(msg=f"path: {path}"):
                # make sure to change log_dir, so tests do not create empty log folders
                cfg = fill_in_defaults(
                    {"log_dir": f"./tests/test_data/TEST_loader/{path.split('.')[0]}/"},
                    load_config(path),
                )
                self.assertIsInstance(cfg, dict)
                self.assertIn("name", cfg)
                name = cfg["name"]
                cfg_w_def = fill_in_defaults(cfg, default_cfg=default_cfg)
                b = BaseModule(cfg_w_def, [])
                self.assertIsInstance(b, BaseModule)
                self.assertEqual(b.name, name)
        shutil.rmtree("./tests/test_data/TEST_loader/", ignore_errors=True)

    def test_load_config_exception(self):
        for fp in [
            "./tests/test_data/other_config.yaml",
            "./test_data/configs/test_config.yaml",
        ]:
            with self.assertRaises(InvalidPathException):
                _ = load_config(fp)


class TestInsertIntoConfig(unittest.TestCase):
    def test_insert_into_config(self):
        test_cfg = get_test_config()

        for path, value, default, copy, result in [
            # empty path
            ([], {"override": False}, {}, False, {"override": False}),
            ([], {"override": False}, {}, True, {"override": False}),
            ([], {"override": True}, {"override": False}, False, {"override": True}),
            ([], {"override": True}, {"override": False}, True, {"override": True}),
            ([], {}, None, True, {}),
            # one nested deep
            (["test"], {"override": False}, {}, False, {"test": {"override": False}}),
            (["test"], {"override": False}, {}, True, {"test": {"override": False}}),
            (["test"], {"override": True}, {"test": 0}, False, {"test": {"override": True}}),
            (["test"], {"override": True}, {"test": 0}, True, {"test": {"override": True}}),
            # test config
            (
                [],
                {"override": False},
                deepcopy(test_cfg),
                True,
                fill_in_defaults({"override": False}, deepcopy(test_cfg)),
            ),
            (
                ["device"],
                {"override": True},
                deepcopy(test_cfg),
                True,
                fill_in_defaults({"device": {"override": True}}, deepcopy(test_cfg)),
            ),
            # deeply nested
            (
                ["bar", "deeper", "pickaxe", "copper"],
                {"override": False},
                {},
                False,
                {"bar": {"deeper": {"pickaxe": {"copper": {"override": False}}}}},
            ),
            (
                ["bar", "deeper", "pickaxe", "copper"],
                {"override": False},
                deepcopy(NESTED_CONFIG),
                False,
                fill_in_defaults({"bar": {"deeper": {"pickaxe": {"copper": {"override": False}}}}}, NESTED_CONFIG),
            ),
            (
                ["bar", "deeper", "pickaxe", "copper"],
                {"override": False},
                deepcopy(NESTED_CONFIG),
                True,
                fill_in_defaults({"bar": {"deeper": {"pickaxe": {"copper": {"override": False}}}}}, NESTED_CONFIG),
            ),
            (
                ["bar", "deeper", "ore"],
                {"override": True},
                deepcopy(NESTED_CONFIG),
                False,
                fill_in_defaults({"bar": {"deeper": {"ore": {"override": True}}}}, NESTED_CONFIG),
            ),
            (
                ["bar", "deeper", "ore"],
                {"override": True},
                deepcopy(NESTED_CONFIG),
                True,
                fill_in_defaults({"bar": {"deeper": {"ore": {"override": True}}}}, NESTED_CONFIG),
            ),
        ]:
            with self.subTest(msg=f"path: {path}, value: {value}, default: {default}, copy: {copy}, result: {result}"):
                orig = deepcopy(default)
                r = insert_into_config(path, value, original=default, copy=copy)
                self.assertEqual(r, result)
                # if copy is true, default should stay the same
                # otherwise there should be updated values in default
                if copy:
                    self.assertEqual(orig, default)
                else:
                    self.assertEqual(get_sub_config(default, path), value)
        self.assertNotIn("override", test_cfg)


if __name__ == "__main__":
    unittest.main()
