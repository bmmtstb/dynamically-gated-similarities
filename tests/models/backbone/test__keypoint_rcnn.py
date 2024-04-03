import os
import unittest

import torch
from torch.nn import Module as TorchModule

from dgs.models.backbone.backbone import BaseBackboneModule
from dgs.models.backbone.keypoint_rcnn import KeypointRCNNBackbone
from dgs.utils.state import State
from helper import get_test_config

IMAGE_PATH = os.path.abspath(os.path.join("./tests/test_data/images/", "torch_person.jpg"))


class TestKPRCNNModel(unittest.TestCase):

    def test_init(self):
        cfg = get_test_config()
        m = KeypointRCNNBackbone(config=cfg, path=[])
        self.assertTrue(isinstance(m, BaseBackboneModule))
        self.assertTrue(isinstance(m.model, TorchModule))
        self.assertFalse(m.model.training)

    def test_forward(self):
        cfg = get_test_config()
        m = KeypointRCNNBackbone(config=cfg, path=[])

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


if __name__ == "__main__":
    unittest.main()
