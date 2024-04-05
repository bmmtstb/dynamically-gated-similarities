import os
import unittest

import torch
from torch.nn import Module as TorchModule

from dgs.models.backbone.backbone import BaseBackboneModule
from dgs.models.backbone.keypoint_rcnn import KeypointRCNNBackbone
from dgs.utils.config import fill_in_defaults
from dgs.utils.state import State
from helper import get_test_config

IMAGE_PATH = os.path.abspath(os.path.join("./tests/test_data/images/", "torch_person.jpg"))


class TestKPRCNNModel(unittest.TestCase):

    def test_init(self):
        cfg = fill_in_defaults({"device": "cuda" if torch.cuda.is_available() else "cpu"}, get_test_config())
        m = KeypointRCNNBackbone(config=cfg, path=[])
        self.assertTrue(isinstance(m, BaseBackboneModule))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)
        self.assertTrue(m.threshold > 0.0)

    def test_forward(self):
        cfg = fill_in_defaults(
            {"device": "cuda" if torch.cuda.is_available() else "cpu", "kprcnn": {"threshold": 0.0}}, get_test_config()
        )
        m = KeypointRCNNBackbone(config=cfg, path=["kprcnn"])

        for fp, B in [
            (IMAGE_PATH, 2),
            ((IMAGE_PATH, IMAGE_PATH), 4),
        ]:
            with self.subTest(msg="fp: {}, B: {}".format(fp, B)):
                out = m(fp)
                self.assertTrue(isinstance(out, State))
                self.assertEqual(out.B, B)
                self.assertFalse(out.validate)
                self.assertEqual(out.keypoints.shape, torch.Size((B, 17, 2)))
                self.assertEqual(out.joint_weight.shape, torch.Size((B, 17, 1)))
                self.assertEqual(len(out.filepath), B)

    def test_forward_with_threshold(self):
        cfg = fill_in_defaults(
            {"device": "cuda" if torch.cuda.is_available() else "cpu", "kprcnn": {"threshold": 0.5}}, get_test_config()
        )

        m = KeypointRCNNBackbone(config=cfg, path=["kprcnn"])
        out = m(IMAGE_PATH)
        self.assertTrue(isinstance(out, State))
        self.assertEqual(out.B, 1)


if __name__ == "__main__":
    unittest.main()
