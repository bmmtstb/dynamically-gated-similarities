import unittest

import torch
from torchvision.tv_tensors import BoundingBoxes

from dgs.utils.state import collate_bboxes, State
from dgs.utils.track import Track
from helper import test_multiple_devices

MAX_LENGTH: int = 30

DUMMY_BBOX: BoundingBoxes = BoundingBoxes([0, 0, 1, 1], format="XYXY", canvas_size=(100, 100), device="cpu")

DUMMY_STATE: State = State(bbox=DUMMY_BBOX)
DUMMY_STATES: list[State] = [
    State(bbox=BoundingBoxes([0, 0, i, i], format="XYXY", canvas_size=(100, 100), device="cpu"))
    for i in range(MAX_LENGTH)
]

EMPTY_TRACK: Track = Track(N=MAX_LENGTH)
ONE_TRACK: Track = Track(N=MAX_LENGTH, states=[DUMMY_STATE])
FULL_TRACK: Track = Track(N=MAX_LENGTH, states=DUMMY_STATES)


class TestTrack(unittest.TestCase):

    def test_init(self):
        t0 = EMPTY_TRACK.copy()
        self.assertEqual(len(t0), 0)
        self.assertEqual(t0.N, MAX_LENGTH)
        self.assertEqual(t0.B, 1)
        self.assertEqual(t0.id, -1)

        t1 = ONE_TRACK.copy()
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1.N, MAX_LENGTH)
        self.assertEqual(t1.B, 1)
        self.assertEqual(t1.id, -1)

        with self.assertRaises(ValueError) as e:
            _ = Track(N=MAX_LENGTH, B=2, states=[DUMMY_STATE])
        self.assertTrue("must have the same shape as the given shape '2'." in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = Track(N=1, B=-1)
        self.assertTrue("B must be greater than 0 but got" in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = Track(N=-1)
        self.assertTrue("N must be greater than 0 but got" in str(e.exception), msg=e.exception)

    def test_get_item(self):
        t0 = EMPTY_TRACK.copy()
        with self.assertRaises(IndexError):
            _ = t0[0]

        t1 = FULL_TRACK.copy()
        for i in range(MAX_LENGTH):
            self.assertEqual(t1[i].bbox[0, 2], i)

    def test_equal(self):
        for t1, t2, eq in [
            (EMPTY_TRACK, EMPTY_TRACK, True),
            (ONE_TRACK, ONE_TRACK, True),
            (FULL_TRACK, FULL_TRACK, True),
            (ONE_TRACK, FULL_TRACK, False),
            (ONE_TRACK, EMPTY_TRACK, False),
            (EMPTY_TRACK, "dummy", False),
            (Track(N=5, states=[DUMMY_STATE]), Track(N=5, states=[DUMMY_STATE, DUMMY_STATE]), False),
            (Track(N=5, states=[], tid=1), Track(N=5, states=[], tid=1), True),
            (Track(N=5, states=[], tid=1), Track(N=5, states=[], tid=2), False),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, eq: {}".format(t1, t2, eq)):
                self.assertEqual(t1 == t2, eq)

    def test_len(self):
        for t, length in [
            (EMPTY_TRACK, 0),
            (ONE_TRACK, 1),
            (FULL_TRACK, MAX_LENGTH),
        ]:
            with self.subTest(msg="t: {}, length: {}".format(t, length)):
                self.assertEqual(len(t), length)


class TestTrackProperties(unittest.TestCase):
    def test_device(self):
        with self.assertRaises(ValueError) as e:
            _ = EMPTY_TRACK.device
        self.assertTrue("Can not get the device of an empty Track" in str(e.exception), msg=e.exception)

        self.assertEqual(ONE_TRACK.device, torch.device("cpu"))

    def test_B(self):
        self.assertEqual(EMPTY_TRACK.B, 1)
        self.assertEqual(ONE_TRACK.B, 1)
        self.assertEqual(FULL_TRACK.B, 1)

        self.assertEqual(Track(N=MAX_LENGTH, B=2).B, 2)

        self.assertEqual(
            Track(
                N=MAX_LENGTH,
                B=2,
                states=[State(bbox=BoundingBoxes(torch.ones((2, 4)), format="XYXY", canvas_size=(100, 100)))],
            ).B,
            2,
        )

    def test_N(self):
        self.assertEqual(EMPTY_TRACK.N, MAX_LENGTH)
        self.assertEqual(ONE_TRACK.N, MAX_LENGTH)
        self.assertEqual(FULL_TRACK.N, MAX_LENGTH)

        self.assertEqual(Track(N=2).N, 2)

    def test_id(self):
        for val_id, res_id in [
            (2, 2),
            (torch.tensor(2), 2),
            (torch.tensor([2]), 2),
        ]:
            with self.subTest(msg="val_id: {}, res_id: {}".format(val_id, res_id)):
                t = ONE_TRACK.copy()
                self.assertEqual(t.id, -1)
                t.id = val_id
                self.assertEqual(t.id, res_id)

        t = ONE_TRACK.copy()
        with self.assertRaises(NotImplementedError) as e:
            t.id = "2"
        self.assertTrue(
            f"unknown type for ID, expected int but got '{type('2')}' - '2'" in str(e.exception), msg=e.exception
        )


class TestTrackFunctions(unittest.TestCase):

    def test_append(self):
        t = Track(N=MAX_LENGTH)
        for i in range(MAX_LENGTH + 3):
            t.append(State(bbox=BoundingBoxes([0, 0, i, i], format="XYXY", canvas_size=(100, 100))))
            self.assertEqual(len(t), min(i + 1, MAX_LENGTH))
            self.assertTrue(torch.allclose(t[-1].bbox.data, torch.tensor([0, 0, i, i])))

    def test_append_wrong_shape(self):
        t = ONE_TRACK.copy()
        with self.assertRaises(ValueError) as e:
            t.append(State(bbox=BoundingBoxes(torch.ones((2, 4)), format="XYXY", canvas_size=(100, 100))))
        self.assertTrue("only get a State with the same batch size of B (1)," in str(e.exception), msg=e.exception)

    def test_clear(self):
        empty = EMPTY_TRACK.copy()
        empty.clear()
        self.assertEqual(empty, EMPTY_TRACK)

        full = FULL_TRACK.copy()
        self.assertEqual(len(full), MAX_LENGTH)
        full.clear()
        self.assertEqual(len(full), 0)
        self.assertEqual(full, EMPTY_TRACK)

    @test_multiple_devices
    def test_to(self, device):
        q = ONE_TRACK.copy()
        self.assertEqual(q.device.type, "cpu")
        q.to(device=device)
        self.assertEqual(q[0].device, device)
        self.assertEqual(q.device, device)

    @test_multiple_devices
    def test_get_all(self, device):
        empty_q = Track(N=MAX_LENGTH)
        with self.assertRaises(ValueError) as e:
            _ = empty_q.get_all()
        self.assertTrue("Can not stack the items of an empty Track." in str(e.exception), msg=e.exception)

        q = FULL_TRACK.copy()
        stacked = q.get_all()
        self.assertTrue(isinstance(stacked, State))
        self.assertEqual(len(stacked), MAX_LENGTH)
        self.assertTrue(
            torch.allclose(
                stacked.bbox,
                State(
                    bbox=collate_bboxes(
                        [BoundingBoxes([0, 0, i, i], format="XYXY", canvas_size=(100, 100)) for i in range(MAX_LENGTH)]
                    )
                ).bbox,
            )
        )
        self.assertTrue(stacked.device, device)

    def test_get_device_on_empty(self):
        q = EMPTY_TRACK.copy()
        with self.assertRaises(ValueError) as e:
            _ = q.device
        self.assertTrue("Can not get the device of an empty Track" in str(e.exception), msg=e.exception)

    def setUp(self):
        self.assertEqual(len(EMPTY_TRACK), 0)
        self.assertEqual(len(ONE_TRACK), 1)
        self.assertEqual(len(FULL_TRACK), MAX_LENGTH)

    def tearDown(self):
        self.assertEqual(len(EMPTY_TRACK), 0)
        self.assertEqual(len(ONE_TRACK), 1)
        self.assertEqual(len(FULL_TRACK), MAX_LENGTH)


if __name__ == "__main__":
    unittest.main()
