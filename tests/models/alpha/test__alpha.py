import unittest
from unittest.mock import patch

import torch as t

from dgs.models.alpha.alpha import BaseAlphaModule
from dgs.models.alpha.fully_connected import FullyConnectedAlpha
from dgs.utils.config import insert_into_config
from helper import get_test_config


class TestBaseAlphaModule(unittest.TestCase):

    default_path = ["alpha"]
    default_cfg = insert_into_config(
        path=default_path,
        value={
            "module_name": "FullyConnectedAlpha",
            "name": "bbox",
            "hidden_layers": [4, 1],
            "bias": True,
            "act_func": "Sigmoid",
        },
        original=get_test_config(),
    )

    @patch.multiple(BaseAlphaModule, __abstractmethods__=set())
    def test_init(self):
        m = BaseAlphaModule(config=self.default_cfg, path=self.default_path)
        self.assertTrue(isinstance(m, BaseAlphaModule))
        self.assertEqual(m.module_type, "alpha")

    @patch.multiple(BaseAlphaModule, __abstractmethods__=set())
    def test_sub_forward(self):
        m = BaseAlphaModule(config=self.default_cfg, path=self.default_path)
        ones = t.ones((3, 2))
        # model not set
        self.assertTrue(t.allclose(m.sub_forward(ones), ones))

        m.model = t.nn.Identity()
        self.assertTrue(t.allclose(m.sub_forward(ones), ones))

    @patch.multiple(BaseAlphaModule, __abstractmethods__=set())
    def test_init_weights_empty(self):
        m = FullyConnectedAlpha(config=self.default_cfg, path=self.default_path)
        self.assertTrue(isinstance(m.model, t.nn.Module))
        self.assertTrue(len(list(m.model.parameters())) > 0)

    @patch.multiple(BaseAlphaModule, __abstractmethods__=set())
    def test_init_weights(self):
        cfg = insert_into_config(
            path=self.default_path,
            value={"weight": "./tests/test_data/weights/fully_connected_alpha.pth"},
            original=self.default_cfg.copy(),
        )
        m = FullyConnectedAlpha(config=cfg, path=self.default_path)

        w = t.load("./tests/test_data/weights/fully_connected_alpha.pth", map_location="cpu")

        self.assertEqual(len(m.model.state_dict()), len(w))
        self.assertEqual(m.model.state_dict().keys(), w.keys())

        for msd, wsd in zip(m.model.state_dict().values(), w.values()):
            self.assertTrue(t.allclose(msd, wsd))


if __name__ == "__main__":
    unittest.main()
