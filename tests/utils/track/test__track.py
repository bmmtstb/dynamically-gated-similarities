import unittest

import torch as t
from torchvision.tv_tensors import BoundingBoxes

from dgs.utils.state import collate_bboxes, State
from dgs.utils.track import Track, TrackStatus
from helper import test_multiple_devices

MAX_LENGTH: int = 30

DUMMY_BBOX: BoundingBoxes = BoundingBoxes([0, 0, 1, 1], format="XYXY", canvas_size=(100, 100), device="cpu")

DUMMY_STATE: State = State(bbox=DUMMY_BBOX)
DUMMY_STATES: list[State] = [
    State(bbox=BoundingBoxes([0, 0, i, i], format="XYXY", canvas_size=(100, 100), device="cpu"))
    for i in range(MAX_LENGTH)
]

EMPTY_TRACK: Track = Track(N=MAX_LENGTH, curr_frame=0)
ONE_TRACK: Track = Track(N=MAX_LENGTH, curr_frame=0, states=[DUMMY_STATE])
FULL_TRACK: Track = Track(N=MAX_LENGTH, curr_frame=0, states=DUMMY_STATES)


class TestTrack(unittest.TestCase):

    def test_init(self):
        t0 = EMPTY_TRACK.copy()
        self.assertEqual(len(t0), 0)
        self.assertEqual(t0.N, MAX_LENGTH)
        self.assertEqual(t0.id, -1)
        self.assertEqual(t0.status, TrackStatus.New)
        self.assertEqual(t0._start_frame, 0)

        t1 = ONE_TRACK.copy()
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1.N, MAX_LENGTH)
        self.assertEqual(t1.id, -1)
        self.assertEqual(t0.status, TrackStatus.New)
        self.assertEqual(t0._start_frame, 0)

        with self.assertRaises(ValueError) as e:
            _ = Track(
                N=MAX_LENGTH,
                curr_frame=0,
                states=[State(bbox=BoundingBoxes(t.ones(2, 4), format="XYWH", canvas_size=(10, 10)))],
            )
        self.assertTrue("The batch size of all the States '[2]' must be 1." in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = Track(N=-1, curr_frame=0)
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
            (EMPTY_TRACK, "dummy", False),  # non Track
            (  # length of states
                Track(N=5, curr_frame=0, states=[DUMMY_STATE]),
                Track(N=5, curr_frame=0, states=[DUMMY_STATE, DUMMY_STATE]),
                False,
            ),
            # tid
            (Track(N=5, curr_frame=0, tid=1), Track(N=5, curr_frame=0, tid=1), True),
            (Track(N=5, curr_frame=0, tid=1), Track(N=5, curr_frame=0, tid=2), False),
            (
                Track(N=5, curr_frame=0, tid=1, states=[DUMMY_STATE]),
                Track(N=5, curr_frame=0, tid=2, states=[DUMMY_STATE]),
                False,
            ),
            # current frame
            (Track(N=5, curr_frame=1), Track(N=5, curr_frame=0), False),
            (Track(N=5, curr_frame=0, states=[DUMMY_STATE]), Track(N=5, curr_frame=1, states=[DUMMY_STATE]), False),
        ]:
            with self.subTest(msg="t1: {}, t2: {}, eq: {}, start: {}, N: {}".format(t1, t2, eq, t1._start_frame, t1.N)):
                self.assertEqual(t1 == t2, eq)

        # test equality with unequal nof active frames
        t1 = Track(N=5, curr_frame=0)
        t1.set_active()
        t2 = Track(N=5, curr_frame=0)
        self.assertFalse(t1 == t2, "Active Track should not equal new track.")

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

        self.assertEqual(ONE_TRACK.device, t.device("cpu"))

    def test_N(self):
        self.assertEqual(EMPTY_TRACK.N, MAX_LENGTH)
        self.assertEqual(ONE_TRACK.N, MAX_LENGTH)
        self.assertEqual(FULL_TRACK.N, MAX_LENGTH)

        self.assertEqual(Track(N=2, curr_frame=0).N, 2)

    def test_id(self):
        for val_id, res_id in [
            (2, 2),
            (t.tensor(2), 2),
            (t.tensor([2]), 2),
        ]:
            with self.subTest(msg="val_id: {}, res_id: {}".format(val_id, res_id)):
                tr = ONE_TRACK.copy()
                self.assertEqual(tr.id, -1)
                tr.id = val_id
                self.assertEqual(tr.id, res_id)

        tr = ONE_TRACK.copy()
        with self.assertRaises(NotImplementedError) as e:
            tr.id = "2"
        self.assertTrue(
            f"unknown type for ID, expected int but got '{type('2')}' - '2'" in str(e.exception), msg=e.exception
        )


class TestTrackFunctions(unittest.TestCase):

    def test_append(self):
        tr = Track(N=MAX_LENGTH, curr_frame=0)
        self.assertEqual(tr.status, TrackStatus.New)

        for i in range(MAX_LENGTH + 3):
            tr.append(State(bbox=BoundingBoxes([0, 0, i, i], format="XYXY", canvas_size=(100, 100))))
            self.assertEqual(len(tr), min(i + 1, MAX_LENGTH))
            self.assertTrue(t.allclose(tr[-1].bbox.data, t.tensor([0, 0, i, i])))
            self.assertEqual(tr.nof_active, i + 1)
            self.assertEqual(tr.status, TrackStatus.Active)

    def test_append_wrong_shape(self):
        tr = ONE_TRACK.copy()
        with self.assertRaises(ValueError) as e:
            tr.append(State(bbox=BoundingBoxes(t.ones((2, 4)), format="XYXY", canvas_size=(100, 100))))
        self.assertTrue("only get a State with the a batch size of 1, but got 2" in str(e.exception), msg=e.exception)
        # status should not have changed
        self.assertEqual(tr.status, TrackStatus.New)

    def test_clear(self):
        empty = EMPTY_TRACK.copy()
        empty.clear()
        self.assertEqual(empty, EMPTY_TRACK)

        full = FULL_TRACK.copy()
        self.assertEqual(len(full), MAX_LENGTH)
        full.clear()
        self.assertEqual(len(full), 0)
        self.assertEqual(full, EMPTY_TRACK)

    def test_status(self):
        tr = Track(N=MAX_LENGTH, curr_frame=0, tid=1)
        self.assertEqual(tr.status, TrackStatus.New)
        self.assertEqual(tr.id, 1)

        tr.append(DUMMY_STATE)
        self.assertEqual(tr.status, TrackStatus.Active)
        self.assertEqual(tr.nof_active, 1)
        self.assertEqual(tr.id, 1)

        tr.set_inactive()
        self.assertEqual(tr.status, TrackStatus.Inactive)
        self.assertEqual(tr.nof_active, 0)
        self.assertEqual(tr.id, 1)

        tr.set_active()
        self.assertEqual(tr.status, TrackStatus.Active)
        self.assertEqual(tr.nof_active, 0)
        self.assertEqual(tr.id, 1)

        tr.append(DUMMY_STATE)
        self.assertEqual(tr.status, TrackStatus.Active)
        self.assertEqual(tr.nof_active, 1)
        self.assertEqual(tr.id, 1)

        tr.set_removed()
        self.assertEqual(tr.status, TrackStatus.Removed)
        self.assertEqual(tr.nof_active, 0)
        self.assertEqual(tr.id, -1)

    def test_set_status(self):
        for status, tid in [
            (TrackStatus.New, ...),
            (TrackStatus.Active, ...),
            (TrackStatus.Inactive, ...),
            (TrackStatus.Reactivated, 0),
            (TrackStatus.Removed, ...),
        ]:
            with self.subTest(msg="status: {}, tid: {}".format(status, tid)):
                tr = ONE_TRACK.copy()
                tr.set_status(status, tid=tid)
                self.assertEqual(tr.status, status)

    def test_age(self):
        tr = Track(N=MAX_LENGTH, curr_frame=0)
        self.assertEqual(tr.age(10), 10)

    @test_multiple_devices
    def test_to(self, device):
        q = ONE_TRACK.copy()
        self.assertEqual(q.device.type, "cpu")
        q.to(device=device)
        self.assertEqual(q[0].device, device)
        self.assertEqual(q.device, device)

    @test_multiple_devices
    def test_get_all(self, device):
        empty_q = Track(N=MAX_LENGTH, curr_frame=0)
        with self.assertRaises(ValueError) as e:
            _ = empty_q.get_all()
        self.assertTrue("Can not stack the items of an empty Track." in str(e.exception), msg=e.exception)

        q = FULL_TRACK.copy()
        stacked = q.get_all()
        self.assertTrue(isinstance(stacked, State))
        self.assertEqual(len(stacked), MAX_LENGTH)
        self.assertTrue(
            t.allclose(
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
        self.assertEqual(EMPTY_TRACK.status, TrackStatus.New)
        self.assertEqual(ONE_TRACK.status, TrackStatus.New)
        self.assertEqual(FULL_TRACK.status, TrackStatus.New)

    def tearDown(self):
        self.assertEqual(len(EMPTY_TRACK), 0)
        self.assertEqual(len(ONE_TRACK), 1)
        self.assertEqual(len(FULL_TRACK), MAX_LENGTH)
        self.assertEqual(EMPTY_TRACK.status, TrackStatus.New)
        self.assertEqual(ONE_TRACK.status, TrackStatus.New)
        self.assertEqual(FULL_TRACK.status, TrackStatus.New)


if __name__ == "__main__":
    unittest.main()
