import unittest

from dgs.utils.config import get_sub_config

EMPTY_CONFIG = {}
SIMPLE_CONFIG = {"foo": "foo foo"}

NESTED_CONFIG = {
    "foo": 3,
    "bar": {
        "x": 1,
        "y": 2,
        "deeper": {
            "ore": "iron",
            "pickaxe": {"iron": 10, "diamnon": 100},
        },
    },
    "dog": {"charlie": "good", "leika": "bad"},
}


class TestGetSubConfig(unittest.TestCase):
    def test_one_deep(self):
        for in_config, path, out_config in [
            (EMPTY_CONFIG, [], EMPTY_CONFIG),
            (SIMPLE_CONFIG, ["foo"], "foo foo"),
            (NESTED_CONFIG, ["foo"], 3),
            (NESTED_CONFIG, ["dog"], {"charlie": "good", "leika": "bad"}),
        ]:
            with self.subTest(msg=f"Path: {path}"):
                self.assertEqual(get_sub_config(in_config, path), out_config)

    def test_recursive(self):
        for in_config, path, out_config in [
            (NESTED_CONFIG, ["bar", "x"], 1),
            (NESTED_CONFIG, ["bar", "y"], 2),
            (NESTED_CONFIG, ["bar", "deeper", "ore"], "iron"),
            (NESTED_CONFIG, ["bar", "deeper", "pickaxe", "iron"], 10),
            (NESTED_CONFIG, ["dog", "charlie"], "good"),
        ]:
            with self.subTest(msg=f"Path: {path}"):
                self.assertEqual(get_sub_config(in_config, path), out_config)

    def test_exceptions(self):
        for config, path in [
            (EMPTY_CONFIG, ["foo"]),
            (SIMPLE_CONFIG, ["bar"]),
            (SIMPLE_CONFIG, ["foo", "bar"]),
            (NESTED_CONFIG, ["foo", "bar"]),
            (NESTED_CONFIG, ["bar", "charlie"]),
        ]:
            with self.subTest(msg=f"Path: {path}"):
                with self.assertRaises(KeyError):
                    get_sub_config(config, path)


if __name__ == "__main__":
    unittest.main()