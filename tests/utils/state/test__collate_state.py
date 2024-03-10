import unittest

import torch
from torchvision.tv_tensors import BoundingBoxes, Image

from dgs.utils.state import collate_bboxes, collate_devices, collate_states, collate_tensors, collate_tvt_tensors, State

N = 10
J = 17
j_dim = 2
C, H, W = (3, 256, 512)


class TestCollate(unittest.TestCase):

    def test_bbox(self):
        bbox = BoundingBoxes(torch.ones(4), format="XYWH", canvas_size=(100, 100))
        for bboxes, result in [
            ([bbox], bbox),
            (
                [bbox for _ in range(N)],
                BoundingBoxes(torch.ones((N, 4)), format="XYXY", canvas_size=(100, 100)),
            ),
        ]:
            with self.subTest(msg="bboxes: {}, result: {}".format(bboxes, result)):
                self.assertTrue(torch.allclose(collate_bboxes(bboxes), result), bboxes)

    def test_devices(self):
        for devices, result in [
            (["cpu"], "cpu"),
            (["cpu", "cuda"], "cpu"),
            ([torch.device("cpu") for _ in range(N)], torch.device("cpu")),
        ]:
            with self.subTest(msg="devices: {}, result: {}".format(devices, result)):
                self.assertEqual(collate_devices(devices), result)

    def test_tvt_tensors(self):
        for tensors, result in [
            ([Image(torch.ones((C, H, W)))], Image(torch.ones((1, C, H, W)))),
            ([Image(torch.ones((1, C, H, W)))], Image(torch.ones((1, C, H, W)))),
            ([Image(torch.ones((C, H, W))) for _ in range(N)], Image(torch.ones((N, C, H, W)))),
            ([Image(torch.ones((1, C, H, W))) for _ in range(N)], Image(torch.ones((N, C, H, W)))),
        ]:
            with self.subTest(
                msg="tensors: {}, t0: {} result: {}".format(len(tensors), tensors[0].shape, result.shape)
            ):
                self.assertTrue(torch.allclose(collate_tvt_tensors(tensors), result), tensors)

    def test_tensors(self):
        for tensors, result in [
            ([torch.ones((1, 11))], torch.ones((1, 11))),
            ([torch.ones((1, 11)) for _ in range(N)], torch.ones((N, 11))),
            ([torch.ones((2, 11)) for _ in range(N)], torch.ones((N, 2, 11))),
        ]:
            with self.subTest(msg="tensors: {}, result: {}".format(len(tensors), result.shape)):
                self.assertTrue(torch.allclose(collate_tensors(tensors), result), tensors)

    def test_states(self):
        bbox = BoundingBoxes(torch.ones(4), format="XYWH", canvas_size=(100, 100))
        s = State(bbox=bbox, keypoints=torch.ones(1, J, j_dim), image=Image(torch.ones(1, C, H, W)))
        for states, result in [
            ([s], s),
            (s, s),
            (
                [s for _ in range(N)],
                State(
                    bbox=BoundingBoxes(torch.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
                    keypoints=torch.ones((N, J, j_dim)),
                    image=Image(torch.ones(N, C, H, W)),
                ),
            ),
        ]:
            with self.subTest(msg="states: {}, result: {}".format(len(states), result.B)):
                self.assertTrue(collate_states(states) == result)


if __name__ == "__main__":
    unittest.main()
