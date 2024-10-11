import unittest

import torch as t

from dgs.models.alpha.combined import SequentialCombinedAlpha
from dgs.utils.config import insert_into_config
from dgs.utils.state import State
from helper import get_test_config
from utils.state import B, DUMMY_BBOX_BATCH


class TestSequentialCombinedAlphaModule(unittest.TestCase):

    def setUp(self):
        self.sequential_name = "sequential"
        self.first_path = ["alpha"]
        self.second_path = ["small"]
        self.default_cfg = get_test_config().copy()
        self.default_cfg[self.first_path[0]] = {
            "module_name": "FullyConnectedAlpha",
            "name": "bbox",
            "hidden_layers": [4, 2],
            "bias": False,
        }
        self.default_cfg[self.second_path[0]] = {
            "module_name": "FullyConnectedAlpha",
            "hidden_layers": [2, 1],
            "bias": False,
        }
        self.default_cfg[self.sequential_name] = {
            "module_name": "SequentialCombinedAlpha",
            "paths": [self.first_path, self.second_path],
        }

        self.m = SequentialCombinedAlpha(path=[self.sequential_name], config=self.default_cfg)

    def tearDown(self):
        del self.default_cfg

    def test_init(self):
        self.assertTrue(isinstance(self.m, SequentialCombinedAlpha))
        self.assertTrue(hasattr(self.m, "model"))
        self.assertTrue(isinstance(self.m.model, t.nn.Sequential))
        self.assertEqual(len(self.m.model), 2)

    def test_init_fails_without_first_module_name(self):
        cfg = insert_into_config(
            path=["invalid"],
            value={"module_name": "SequentialCombinedAlpha", "paths": [self.second_path, self.first_path]},
            original=self.default_cfg,
            copy=True,
        )
        with self.assertRaises(ValueError) as e:
            _ = SequentialCombinedAlpha(path=["invalid"], config=cfg)
        self.assertTrue("Configuration of first module must have `name` key" in str(e.exception), msg=e.exception)

    def test_forward(self):
        s = State(bbox=DUMMY_BBOX_BATCH)
        out = self.m.forward(s)
        self.assertEqual(out.shape, t.Size((B, 1)))


if __name__ == "__main__":
    unittest.main()
