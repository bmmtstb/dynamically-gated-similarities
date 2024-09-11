import unittest

import torch
from torchvision.tv_tensors import BoundingBoxes

from dgs.models.similarity import SimilarityModule
from dgs.models.similarity.pose_similarity import IntersectionOverUnion, ObjectKeypointSimilarity
from dgs.utils.config import fill_in_defaults
from dgs.utils.state import State
from helper import get_test_config


class TestPoseSimilarities(unittest.TestCase):
    default_cfg = get_test_config()

    def test_oks(self):
        cfg = fill_in_defaults(
            {"oks": {"format": "coco", "module_name": "oks", "softmax": False}}, default_cfg=self.default_cfg.copy()
        )
        sim = ObjectKeypointSimilarity(config=cfg, path=["oks"])
        self.assertTrue(isinstance(sim, SimilarityModule))

        J = 17
        bbox1 = BoundingBoxes([1, 1, 2, 2], canvas_size=(1, 1), format="xyxy")
        bbox2 = BoundingBoxes([[1.5, 1.5, 1, 1]], canvas_size=(1, 1), format="CXCYWH")
        bbox3 = BoundingBoxes([[0, 0, 1, 1], [0, 0, 1, 1]], canvas_size=(1, 1), format="xywh")
        bbox4 = BoundingBoxes([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]], canvas_size=(1, 1), format="xyxy")

        ds1 = State(
            filepath=("",),
            bbox=bbox1,
            keypoints=torch.ones((J, 2)),
            validate=False,
            joint_weight=torch.ones(J),
        )
        ds2 = State(
            filepath=("",),
            bbox=bbox2,
            keypoints=torch.ones((1, J, 2)),
            validate=False,
            joint_weight=torch.ones((1, J, 1)),
        )
        ds3 = State(
            filepath=("", ""),
            bbox=bbox3,
            keypoints=torch.zeros((2, J, 2)),
            validate=False,
            joint_weight=torch.stack([torch.zeros(J), torch.ones(J)]),
        )
        ds4 = State(
            filepath=("", "", ""),
            bbox=bbox4,
            keypoints=torch.ones((3, J, 2)),
            validate=False,
            joint_weight=torch.stack([torch.zeros(J), torch.ones(J), torch.ones(J)]),
        )

        oks12 = sim(ds1, ds2)
        oks21 = sim(ds2, ds1)
        oks13 = sim(ds1, ds3)
        oks23 = sim(ds2, ds3)
        oks14 = sim(ds1, ds4)
        oks34 = sim(ds3, ds4)
        oks43 = sim(ds4, ds3)

        self.assertEqual(tuple(oks13.shape), (1, 2))
        self.assertEqual(tuple(oks23.shape), (1, 2))
        self.assertEqual(tuple(oks34.shape), (2, 3))
        self.assertEqual(tuple(oks43.shape), (3, 2))

        self.assertTrue(torch.allclose(oks12, torch.ones(1)))
        self.assertTrue(torch.allclose(oks12, oks21))
        self.assertTrue(torch.allclose(oks13[0, 0], torch.zeros(1)))
        self.assertTrue(torch.allclose(oks13, oks23))

        self.assertTrue(torch.allclose(oks14, torch.tensor([[0, 1, 1]], dtype=torch.float32)), oks14)
        self.assertTrue(torch.allclose(oks34[0], torch.tensor([0, 0, 0], dtype=torch.float32)), oks34)
        self.assertTrue(torch.allclose(oks34[0], oks43[:, 0]), oks34)

    def test_iou(self):
        cfg = fill_in_defaults({"iou": {"module_name": "iou", "softmax": False}}, default_cfg=self.default_cfg.copy())
        sim = IntersectionOverUnion(config=cfg, path=["iou"])
        self.assertTrue(isinstance(sim, SimilarityModule))
        bbox1 = BoundingBoxes([0, 0, 5, 5], canvas_size=(10, 10), format="xyxy")
        bbox2 = BoundingBoxes([1, 1, 5, 5], canvas_size=(10, 10), format="xywh")
        bbox3 = BoundingBoxes([[1, 1, 6, 6]], canvas_size=(10, 10), format="xyxy")
        bbox4 = BoundingBoxes([[0, 0, 5, 5], [1, 1, 6, 6], [2, 2, 7, 7]], canvas_size=(10, 10), format="xyxy")

        J = 17

        ds1 = State(
            filepath=("",),
            bbox=bbox1,
            keypoints=torch.ones((J, 2)),
            validate=False,
        )
        ds2 = State(
            filepath=("",),
            bbox=bbox2,
            keypoints=torch.ones((J, 2)),
            validate=False,
        )
        ds3 = State(
            filepath=("",),
            bbox=bbox3,
            keypoints=torch.zeros((1, J, 2)),
            validate=False,
        )
        ds4 = State(
            filepath=("", "", ""),
            bbox=bbox4,
            keypoints=torch.ones((3, J, 2)),
            validate=False,
        )

        iou12 = sim(ds1, ds2)
        iou13 = sim(ds1, ds3)
        iou23 = sim(ds2, ds3)
        iou24 = sim(ds2, ds4)
        iou42 = sim(ds4, ds2)
        iou44 = sim(ds4, ds4)

        self.assertEqual(list(iou12.shape), [1, 1])
        self.assertTrue(torch.allclose(iou12, torch.tensor(16 / 34)))
        self.assertTrue(torch.allclose(iou13, iou12))
        self.assertEqual(iou23.item(), 1.0)

        self.assertEqual(list(iou24.shape), [1, 3])
        self.assertEqual(list(iou42.shape), [3, 1])
        self.assertTrue(torch.allclose(iou24, torch.tensor([16 / 34, 1.0, 16 / 34])))
        self.assertTrue(torch.allclose(iou24, iou42.T))

        self.assertEqual(list(iou44.shape), [3, 3])
        self.assertTrue(
            torch.allclose(
                iou44, torch.tensor([[1.0, 16 / 34, 9 / 41], [16 / 34, 1.0, 16 / 34], [9 / 41, 16 / 34, 1.0]])
            )
        )


if __name__ == "__main__":
    unittest.main()
