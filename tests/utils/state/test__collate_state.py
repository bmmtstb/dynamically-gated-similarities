import unittest

import torch as t
from torchvision.tv_tensors import BoundingBoxes, Image, TVTensor

from dgs.utils.state import (
    collate_bboxes,
    collate_devices,
    collate_states,
    collate_tensors,
    collate_tvt_tensors,
    EMPTY_STATE,
    State,
)

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
                self.assertTrue(isinstance(r, t.Tensor))
                self.assertEqual(len(r), 0)

    def test_bbox(self):
        bbox = BoundingBoxes(t.ones((1, 4)), format="XYWH", canvas_size=(100, 100))
        other = BoundingBoxes(t.ones((2, 4)), format="XYXY", canvas_size=(50, 50))
        bbox_empty = BoundingBoxes(t.empty((0, 4)), format="XYWH", canvas_size=(100, 100))

        for bboxes, result in [
            ([], BoundingBoxes(t.empty((0, 4)), canvas_size=(0, 0), format="XYXY")),
            ([bbox], bbox),
            ([bbox_empty], bbox_empty),
            ([bbox_empty, bbox_empty], bbox_empty),
            (
                [bbox for _ in range(N)],
                BoundingBoxes(t.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
            ),
            ([bbox, other], BoundingBoxes(t.ones((3, 4)), format="XYWH", canvas_size=(100, 100))),
            ([other, bbox], BoundingBoxes(t.ones((3, 4)), format="XYXY", canvas_size=(50, 50))),
            ([bbox_empty, bbox], bbox),
            ([bbox, bbox_empty], bbox),
        ]:
            with self.subTest(msg="bboxes: {}, result: {}".format(bboxes, result)):
                self.assertTrue(t.allclose(collate_bboxes(bboxes), result), msg=bboxes)

    def test_devices(self):
        for devices, result in [
            (["cpu"], "cpu"),
            (["cpu", "cuda"], "cpu"),
            ([t.device("cpu") for _ in range(N)], t.device("cpu")),
        ]:
            with self.subTest(msg="devices: {}, result: {}".format(devices, result)):
                self.assertEqual(collate_devices(devices), result)

    def test_tvt_tensors(self):
        img = Image(t.ones((C, H, W)))
        img_ = Image(t.ones((1, C, H, W)))
        empty_img = Image(t.ones((0, C, H, W)))
        imgs = Image(t.ones((N, C, H, W)))

        for tensors, result in [
            ([], TVTensor([])),
            ([TVTensor(t.tensor(1))], TVTensor(t.tensor(1))),
            ([TVTensor(t.tensor(1)), TVTensor(t.tensor(1))], TVTensor(t.tensor(1))),
            ([empty_img], TVTensor([])),
            ([empty_img, empty_img], TVTensor([])),
            ([img], img_),
            ([img_], img_),
            ([empty_img, img_, empty_img], img_),
            ([img_], Image(t.ones((1, C, H, W)))),
            ([img for _ in range(N)], img.repeat(N, 1, 1)),
            ([img_ for _ in range(N)], imgs),
        ]:
            with self.subTest(
                msg="tensors: {}, t0: {} result: {}".format(
                    len(tensors), tensors[0].shape if len(tensors) else 0, result.shape
                )
            ):
                self.assertTrue(t.allclose(collate_tvt_tensors(tensors), result), tensors)

    def test_tensors(self):
        empty_t = t.ones(0, dtype=t.long)
        empty_l = t.tensor([], dtype=t.long)
        t_0d = t.tensor(1, dtype=t.long)
        t_1d = t.ones(1, dtype=t.long)
        t_2d = t.ones((1, 11), dtype=t.long)

        for tensors, result in [
            # different stages of empty
            ([], t.empty(0)),
            ([empty_l], t.empty((0, 1), dtype=t.long)),
            ([empty_l, empty_l], t.empty((0, 2), dtype=t.long)),
            ([empty_t, t.ones(0, dtype=t.long), empty_l], t.empty((0, 3), dtype=t.long)),
            # regular plus empty
            ([empty_l, t_0d, empty_l], t_1d),
            ([empty_l, t_1d, empty_l], t_1d),
            ([empty_l, t_2d, empty_l], t_2d),
            ([empty_t, t_0d, empty_t], t_1d),
            ([empty_t, t_1d, empty_t], t_1d),
            ([empty_t, t_2d, empty_t], t_2d),
            (
                [
                    t.empty((0, 11), dtype=t.long),
                    t.ones((1, 11), dtype=t.long),
                    t.ones((0, 11), dtype=t.long),
                ],
                t.ones((1, 11), dtype=t.long),
            ),
            # regular
            ([t_0d], t_0d),
            ([t_1d], t_1d),
            ([t_2d], t_2d),
            ([t_0d, t_0d], t.stack([t_0d, t_0d])),
            ([t_1d, t_1d], t.ones((2, 1), dtype=t.long)),
            ([t_2d, t_2d], t.ones((2, 11), dtype=t.long)),
            (
                [t.ones((3, 11), dtype=t.long), t.ones((3, 11), dtype=t.long)],
                t.ones((6, 11), dtype=t.long),
            ),
            (
                [t.ones((1, 3, 11), dtype=t.long), t.ones((1, 3, 11), dtype=t.long)],
                t.ones((2, 3, 11), dtype=t.long),
            ),
            ([t.ones((1, 11), dtype=t.long) for _ in range(N)], t.ones((N, 11), dtype=t.long)),
            ([t.ones((3, 11), dtype=t.long) for _ in range(N)], t.ones((N * 3, 11), dtype=t.long)),
        ]:
            with self.subTest(msg="t len: {}, result: {}, t: {}".format(len(tensors), result.shape, tensors)):
                self.assertTrue(t.allclose(collate_tensors(tensors), result), tensors)


class TestCollateStates(unittest.TestCase):
    bbox = BoundingBoxes(t.ones(4), format="XYWH", canvas_size=(100, 100))

    def test_collate_states(self):
        for validate in [True, False]:
            s = State(
                bbox=self.bbox,
                keypoints=t.ones(1, J, j_dim),
                image=[Image(t.ones(1, C, H, W))],
                validate=validate,
            )
            n_states = State(
                bbox=BoundingBoxes(t.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
                keypoints=t.ones((N, J, j_dim)),
                image=[Image(t.ones(1, C, H, W)) for _ in range(N)],
                validate=validate,
            )

            for states, result in [
                ([s], s),
                (s, s),
                ([s for _ in range(N)], n_states),
                (
                    [State(bbox=self.bbox, str="dummy", tuple=(1,), validate=validate) for _ in range(N)],
                    State(
                        bbox=BoundingBoxes(t.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
                        str=tuple("dummy" for _ in range(N)),
                        tuple=tuple(1 for _ in range(N)),
                        validate=validate,
                    ),
                ),
            ]:
                with self.subTest(msg="s: {}, res: {}, val: {}".format(len(states), result.B, validate)):
                    self.assertTrue(collate_states(states) == result)
                    self.assertEqual(result.validate, validate)

    def test_empty_states(self):
        s = State(bbox=self.bbox, keypoints=t.ones(1, J, j_dim), image=[Image(t.ones(1, C, H, W))], validate=False)
        n_states = State(
            bbox=BoundingBoxes(t.ones((N, 4)), format="XYWH", canvas_size=(100, 100)),
            keypoints=t.ones((N, J, j_dim)),
            image=[Image(t.ones(1, C, H, W)) for _ in range(N)],
            validate=False,
        )

        for states, result in [
            ([], EMPTY_STATE),
            ([EMPTY_STATE], EMPTY_STATE),
            (EMPTY_STATE, EMPTY_STATE),
            ([EMPTY_STATE, EMPTY_STATE], EMPTY_STATE),
            ([EMPTY_STATE, s], s),
            ([v for sublist in [[s, EMPTY_STATE] for _ in range(N)] for v in sublist], n_states),  # [s,ES,s,ES,...]
        ]:
            with self.subTest(msg="s: {}, res: {}".format(len(states), result.B)):
                self.assertTrue(collate_states(states) == result)


if __name__ == "__main__":
    unittest.main()
