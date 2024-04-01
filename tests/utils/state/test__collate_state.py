import unittest

import torch
from torchvision.tv_tensors import BoundingBoxes, Image, TVTensor

from dgs.utils.state import collate_bboxes, collate_devices, collate_states, collate_tensors, collate_tvt_tensors, State

N = 10
J = 17
j_dim = 2
C, H, W = (3, 256, 512)


class TestCollate(unittest.TestCase):

    def test_empty_collate(self):
        data = []
        for func in [collate_bboxes, collate_tensors, collate_tvt_tensors]:
            with self.subTest(msg="func: {}".format(func)):
                r = func(data)
                self.assertTrue(isinstance(r, torch.Tensor))
                self.assertEqual(len(r), 0)

    def test_bbox(self):
        bbox = BoundingBoxes(torch.ones((1, 4)), format="XYWH", canvas_size=(100, 100))
        other = BoundingBoxes(torch.ones((2, 4)), format="XYXY", canvas_size=(50, 50))
        bbox_empty = BoundingBoxes(torch.empty((0, 4)), format="XYWH", canvas_size=(100, 100))

        for bboxes, result in [
            ([], BoundingBoxes(torch.empty((0, 4)), canvas_size=(0, 0), format="XYXY")),
            ([bbox], bbox),
            ([bbox_empty], bbox_empty),
            ([bbox_empty, bbox_empty], bbox_empty),
            (
                [bbox for _ in range(N)],
                BoundingBoxes(torch.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
            ),
            ([bbox, other], BoundingBoxes(torch.ones((3, 4)), format="XYWH", canvas_size=(100, 100))),
            ([other, bbox], BoundingBoxes(torch.ones((3, 4)), format="XYXY", canvas_size=(50, 50))),
            ([bbox_empty, bbox], bbox),
            ([bbox, bbox_empty], bbox),
        ]:
            with self.subTest(msg="bboxes: {}, result: {}".format(bboxes, result)):
                self.assertTrue(torch.allclose(collate_bboxes(bboxes), result), msg=bboxes)

    def test_devices(self):
        for devices, result in [
            (["cpu"], "cpu"),
            (["cpu", "cuda"], "cpu"),
            ([torch.device("cpu") for _ in range(N)], torch.device("cpu")),
        ]:
            with self.subTest(msg="devices: {}, result: {}".format(devices, result)):
                self.assertEqual(collate_devices(devices), result)

    def test_tvt_tensors(self):
        img = Image(torch.ones((C, H, W)))
        img_ = Image(torch.ones((1, C, H, W)))
        empty_img = Image(torch.ones((0, C, H, W)))
        imgs = Image(torch.ones((N, C, H, W)))

        for tensors, result in [
            ([], TVTensor([])),
            ([TVTensor(torch.tensor(1))], TVTensor(torch.tensor(1))),
            ([TVTensor(torch.tensor(1)), TVTensor(torch.tensor(1))], TVTensor(torch.tensor(1))),
            ([empty_img], TVTensor([])),
            ([empty_img, empty_img], TVTensor([])),
            ([img], img_),
            ([img_], img_),
            ([empty_img, img_, empty_img], img_),
            ([img_], Image(torch.ones((1, C, H, W)))),
            ([img for _ in range(N)], img.repeat(N, 1, 1)),
            ([img_ for _ in range(N)], imgs),
        ]:
            with self.subTest(
                msg="tensors: {}, t0: {} result: {}".format(
                    len(tensors), tensors[0].shape if len(tensors) else 0, result.shape
                )
            ):
                self.assertTrue(torch.allclose(collate_tvt_tensors(tensors), result), tensors)

    def test_tensors(self):
        empty_res = torch.empty(0)
        empty_t = torch.ones(0, dtype=torch.long)
        empty_l = torch.tensor([], dtype=torch.long)
        t_0d = torch.tensor(1, dtype=torch.long)
        t_1d = torch.ones(1, dtype=torch.long)
        t_2d = torch.ones((1, 11), dtype=torch.long)

        for tensors, result in [
            # different stages of empty
            ([], empty_res),
            ([empty_l], empty_res),
            ([empty_l, empty_l], empty_res),
            ([empty_t, torch.ones(0, dtype=torch.long), empty_l], empty_res),
            # regular plus empty
            ([empty_l, t_0d, empty_l], t_1d),
            ([empty_l, t_1d, empty_l], t_1d),
            ([empty_l, t_2d, empty_l], t_2d),
            ([empty_t, t_0d, empty_t], t_1d),
            ([empty_t, t_1d, empty_t], t_1d),
            ([empty_t, t_2d, empty_t], t_2d),
            (
                [
                    torch.empty((0, 11), dtype=torch.long),
                    torch.ones((1, 11), dtype=torch.long),
                    torch.ones((0, 11), dtype=torch.long),
                ],
                torch.ones((1, 11), dtype=torch.long),
            ),
            # regular
            ([t_0d], t_0d),
            ([t_1d], t_1d),
            ([t_2d], t_2d),
            ([t_0d, t_0d], torch.stack([t_0d, t_0d])),
            ([t_1d, t_1d], torch.ones((2, 1), dtype=torch.long)),
            ([t_2d, t_2d], torch.ones((2, 11), dtype=torch.long)),
            (
                [torch.ones((3, 11), dtype=torch.long), torch.ones((3, 11), dtype=torch.long)],
                torch.ones((6, 11), dtype=torch.long),
            ),
            (
                [torch.ones((1, 3, 11), dtype=torch.long), torch.ones((1, 3, 11), dtype=torch.long)],
                torch.ones((2, 3, 11), dtype=torch.long),
            ),
            ([torch.ones((1, 11), dtype=torch.long) for _ in range(N)], torch.ones((N, 11), dtype=torch.long)),
            ([torch.ones((3, 11), dtype=torch.long) for _ in range(N)], torch.ones((N * 3, 11), dtype=torch.long)),
        ]:
            with self.subTest(msg="tensors: {}, result: {}".format(len(tensors), result.shape)):
                self.assertTrue(torch.allclose(collate_tensors(tensors), result), tensors)


class TestCollateStates(unittest.TestCase):

    def test_states(self):
        for validate in [True, False]:
            bbox = BoundingBoxes(torch.ones(4), format="XYWH", canvas_size=(100, 100))

            s = State(
                bbox=bbox, keypoints=torch.ones(1, J, j_dim), image=Image(torch.ones(1, C, H, W)), validate=validate
            )

            for states, result in [
                ([s], s),
                (s, s),
                (
                    [s for _ in range(N)],
                    State(
                        bbox=BoundingBoxes(torch.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
                        keypoints=torch.ones((N, J, j_dim)),
                        image=Image(torch.ones(N, C, H, W)),
                        validate=validate,
                    ),
                ),
                (
                    [State(bbox=bbox, str="dummy", tuple=(1,), validate=validate) for _ in range(N)],
                    State(
                        bbox=BoundingBoxes(torch.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
                        str=tuple("dummy" for _ in range(N)),
                        tuple=tuple(1 for _ in range(N)),
                        validate=validate,
                    ),
                ),
            ]:
                with self.subTest(msg="s: {}, res: {}, val: {}".format(len(states), result.B, validate)):
                    self.assertTrue(collate_states(states) == result)
                    self.assertEqual(result.validate, validate)


if __name__ == "__main__":
    unittest.main()
