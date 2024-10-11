import unittest
from unittest.mock import patch

import torch as t

from dgs.models.alpha.alpha import BaseAlphaModule
from dgs.utils.config import insert_into_config
from helper import get_test_config


class TestBaseAlphaModule(unittest.TestCase):

    default_path = ["alpha"]
    default_cfg = insert_into_config(
        path=default_path,
        value={"module_name": "FullyConnectedAlpha", "name": "bbox", "hidden_layers": [4, 1], "bias": False},
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


if __name__ == "__main__":
    unittest.main()
