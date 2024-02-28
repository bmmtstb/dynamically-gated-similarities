import unittest

import torch
from torchvision.tv_tensors import BoundingBoxes

from dgs.models.similarity import SimilarityModule
from dgs.models.similarity.pose_similarity import IntersectionOverUnion, ObjectKeypointSimilarity
from dgs.utils.config import fill_in_defaults
from helper import get_test_config


class TestPoseSimilarities(unittest.TestCase):
    default_cfg = get_test_config()

    def test_oks(self):
        cfg = fill_in_defaults({"oks": {"format": "coco"}}, default_cfg=self.default_cfg.copy())
        sim = ObjectKeypointSimilarity(config=cfg, path=["oks"])
        self.assertTrue(isinstance(sim, SimilarityModule))

        J = 17

        kp1 = torch.ones((J, 2))
        kp2 = torch.ones((1, J, 3))
        kp3 = torch.zeros((J, 2))

        area = torch.tensor(1.0, dtype=torch.float32).unsqueeze(0)

        oks12 = sim((kp1, area), kp2)
        oks13 = sim((kp1, area), kp3)
        oks23 = sim((kp2, area), kp3)

        self.assertTrue(torch.allclose(oks12, torch.ones(1)))
        self.assertTrue(torch.allclose(oks13, oks23))

    def test_iou(self):
        cfg = fill_in_defaults({"iou": {}}, default_cfg=self.default_cfg.copy())
        sim = IntersectionOverUnion(config=cfg, path=["iou"])
        self.assertTrue(isinstance(sim, SimilarityModule))
        box1 = BoundingBoxes([0, 0, 5, 5], canvas_size=(10, 10), format="xyxy")
        box2 = BoundingBoxes([1, 1, 5, 5], canvas_size=(10, 10), format="xywh")
        box3 = BoundingBoxes([1, 1, 6, 6], canvas_size=(10, 10), format="xyxy")

        iou12 = sim(box1, box2)
        iou13 = sim(box1, box3)
        iou23 = sim(box2, box3)

        self.assertTrue(torch.allclose(iou12, torch.tensor(16 / 34)))
        self.assertTrue(torch.allclose(iou13, iou12))
        self.assertEqual(iou23.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
