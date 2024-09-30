import unittest

import torch as t
from torch import nn

from dgs.models.alpha.fully_connected import FullyConnectedAlpha
from dgs.models.loader import module_loader
from dgs.utils.config import fill_in_defaults
from dgs.utils.state import State
from helper import get_test_config
from utils.state import B, DUMMY_DATA, DUMMY_DATA_BATCH


class TestFullyConnectedAlpha(unittest.TestCase):

    default_cfg = fill_in_defaults(
        {
            "default": {"module_name": "FCA", "name": "bbox", "hidden_layers": [4, 1], "bias": False, "act_func": None},
        },
        get_test_config(),
    )

    def test_init(self):
        m = module_loader(config=self.default_cfg, module_type="alpha", key="default")
        self.assertTrue(isinstance(m, FullyConnectedAlpha))
        self.assertTrue(hasattr(m, "model"))
        self.assertTrue(isinstance(m, nn.Module))
        self.assertTrue(hasattr(m, "data_getter"))

    def test_forward(self):
        m: FullyConnectedAlpha = module_loader(config=self.default_cfg, module_type="alpha", key="default")
        # init with 1
        nn.init.constant_(m.model[0].weight, 1.0)

        for s, out in [
            (State(**DUMMY_DATA), t.ones((1, 1)) * 4),
            (State(**DUMMY_DATA_BATCH), t.ones((B, 1)) * 4),
        ]:
            with self.subTest(msg="s: {}, out: {}".format(s, out)):
                r = m.forward(s)
                self.assertTrue(t.allclose(r, out))


if __name__ == "__main__":
    unittest.main()
