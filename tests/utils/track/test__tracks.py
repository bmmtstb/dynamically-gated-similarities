import unittest

import torch
from torchvision.tv_tensors import BoundingBoxes

from dgs.utils.config import DEF_CONF
from dgs.utils.state import State
from dgs.utils.track import Track, Tracks
from helper import test_multiple_devices

THRESH: int = 30

# TRACK
MAX_LENGTH: int = 30

DUMMY_BBOX: BoundingBoxes = BoundingBoxes([0, 0, 1, 1], format="XYXY", canvas_size=(100, 100), device="cpu")
DUMMY_STATE: State = State(bbox=DUMMY_BBOX)
DUMMY_STATES: list[State] = [
    State(bbox=BoundingBoxes([0, 0, i, i], format="XYXY", canvas_size=(100, 100), device="cpu"))
    for i in range(MAX_LENGTH)
]

ONE_TRACK: Track = Track(N=MAX_LENGTH, states=[DUMMY_STATE])
FULL_TRACK: Track = Track(N=MAX_LENGTH, states=DUMMY_STATES)


# TRACKS
EMPTY_TRACKS: Tracks = Tracks(thresh=THRESH)

ONE_TRACKS: Tracks = Tracks(thresh=THRESH)
OT_O_ID = ONE_TRACKS.add(tracks={}, new_tracks=[ONE_TRACK.copy()])[0]

MULTI_TRACKS: Tracks = Tracks(thresh=THRESH)
MT_O_ID = MULTI_TRACKS.add(tracks={}, new_tracks=[ONE_TRACK.copy()])[0]
MT_F_ID = MULTI_TRACKS.add(tracks={0: DUMMY_STATE.copy()}, new_tracks=[FULL_TRACK.copy()])[0]
MULTI_TRACKS.add(tracks={0: DUMMY_STATE.copy()}, new_tracks=[])  # now 1 is inactive !


def _track_w_tid(track: Track, tid: int) -> Track:
    t = track.copy()
    for s in t:
        s.track_id = tid
    return t


class TestTracks(unittest.TestCase):
    def test_init(self):
        t = Tracks(thresh=THRESH)
        self.assertEqual(len(t.data), 0)
        self.assertEqual(len(t.inactive), 0)
        self.assertEqual(t.inactivity_threshold, THRESH)

        # test without explicit thresh
        no_thresh = Tracks()
        self.assertEqual(len(no_thresh.data), 0)
        self.assertEqual(len(no_thresh.inactive), 0)
        self.assertEqual(no_thresh.inactivity_threshold, DEF_CONF.tracks.inactivity_threshold)

        with self.assertRaises(ValueError) as e:
            _ = Tracks(thresh=-1)
        self.assertTrue("Threshold must be positive, got -1." in str(e.exception), msg=e.exception)

    def test_equality(self):
        self.assertTrue(EMPTY_TRACKS == EMPTY_TRACKS)
        self.assertTrue(ONE_TRACKS == ONE_TRACKS)

        self.assertFalse("dummy" == EMPTY_TRACKS)
        self.assertFalse(EMPTY_TRACKS == Tracks(thresh=THRESH - 1))

    def test_copy(self):
        t1 = ONE_TRACKS.copy()
        t = t1.copy()
        t.reset()
        t1.add(tracks={OT_O_ID: DUMMY_STATE.copy()}, new_tracks=[ONE_TRACK.copy()])
        self.assertEqual(len(t), 0)
        self.assertEqual(len(t1), 2)

    def test_reset(self):
        for tracks in [EMPTY_TRACKS, ONE_TRACKS, MULTI_TRACKS]:
            with self.subTest(msg="tracks: {}".format(tracks)):
                t = tracks.copy()
                t.reset()
                self.assertTrue(t == EMPTY_TRACKS)

        self.assertEqual(len(ONE_TRACKS[OT_O_ID]), 1)
        self.assertEqual(len(MULTI_TRACKS[MT_O_ID]), 3)
        self.assertEqual(len(MULTI_TRACKS[MT_F_ID]), 30)

    def test_remove_tid(self):
        t0 = EMPTY_TRACKS.copy()
        t0.remove_tid(0)
        self.assertEqual(len(t0), 0)
        self.assertTrue(t0 == EMPTY_TRACKS)

        t1 = ONE_TRACKS.copy()
        self.assertEqual(len(t1), 1)
        t1.remove_tid(0)
        self.assertEqual(len(t1), 0)
        self.assertTrue(t1 == EMPTY_TRACKS)

        t2 = MULTI_TRACKS.copy()
        self.assertEqual(len(t2), 2)
        t2.remove_tid(MT_F_ID)
        self.assertEqual(len(t2), 1)
        self.assertEqual(len(t2[OT_O_ID]), 3)

    def test_get_next_id(self):
        t0 = EMPTY_TRACKS.copy()
        self.assertEqual(t0._get_next_id(), 0)

        t1 = ONE_TRACKS.copy()
        self.assertEqual(t1._get_next_id(), 1)

    @test_multiple_devices
    def test_to(self, device):
        t = ONE_TRACKS.copy()
        self.assertTrue(t[OT_O_ID].device == torch.device("cpu"))
        t.to(device=device)
        self.assertTrue(t[OT_O_ID].device == device)

    def test_handle_inactive(self):
        t = Tracks(thresh=2)
        tid = t.add({}, new_tracks=[ONE_TRACK.copy()])[0]
        self.assertEqual(t.ids_active(), {tid})
        self.assertEqual(t.ids_inactive(), set())

        t._handle_inactive({tid})
        self.assertEqual(t.ids_active(), set())
        self.assertEqual(t.ids_inactive(), {tid})
        self.assertEqual(t.inactive[tid], 1)

        t._handle_inactive({tid})
        self.assertEqual(t.ids_active(), set())
        self.assertEqual(t.ids_inactive(), set())

    def test_add_track(self):
        t0 = EMPTY_TRACKS.copy()
        r0 = t0._add_track(ONE_TRACK.copy())
        self.assertEqual(r0, 0)
        self.assertTrue(t0.data[r0] == ONE_TRACK)

        r1 = t0._add_track(FULL_TRACK.copy())
        self.assertEqual(r1, 1)
        self.assertTrue(t0.data[r1] == FULL_TRACK)

    def test_update_track(self):
        tracks = ONE_TRACKS.copy()
        self.assertEqual(len(tracks[OT_O_ID]), 1)
        tracks._update_track(OT_O_ID, DUMMY_STATE)
        self.assertEqual(len(tracks[OT_O_ID]), 2)

        with self.assertRaises(KeyError) as e:
            tracks._update_track(5, DUMMY_STATE)
        self.assertTrue("Track-ID 5 not present in Tracks" in str(e.exception), msg=e.exception)

        multi_tracks = MULTI_TRACKS.copy()
        self.assertEqual(multi_tracks.ids_inactive(), {MT_F_ID})
        self.assertEqual(len(multi_tracks[MT_F_ID]), MAX_LENGTH)
        self.assertEqual(multi_tracks.ids_active(), {MT_O_ID})
        multi_tracks._update_track(MT_F_ID, DUMMY_STATE)
        self.assertEqual(multi_tracks.ids_inactive(), set())
        self.assertEqual(multi_tracks.ids_active(), {MT_O_ID, MT_F_ID})
        self.assertEqual(len(multi_tracks[MT_F_ID]), MAX_LENGTH)

    def test_get_item(self):
        empty = EMPTY_TRACKS.copy()
        with self.assertRaises(KeyError) as e:
            _ = empty[0]
        self.assertTrue("0" in str(e.exception), msg=e.exception)

        one = ONE_TRACKS.copy()
        track = one[OT_O_ID]
        self.assertTrue(track == _track_w_tid(ONE_TRACK, OT_O_ID))

    def test_add(self):
        t = Tracks(thresh=2)

        first_tid = t.add(tracks={}, new_tracks=[ONE_TRACK.copy()])[0]
        self.assertEqual(len(t), 1)
        self.assertEqual(t.ids_active(), {first_tid})
        self.assertEqual(t.ids_inactive(), set())
        self.assertTrue(all(s.track_id.item() == first_tid for s in t[first_tid]))

        second_tid = t.add(tracks={0: DUMMY_STATE.copy()}, new_tracks=[FULL_TRACK.copy()])[0]
        self.assertEqual(len(t), 2)
        self.assertEqual(t.ids_active(), {first_tid, second_tid})
        self.assertEqual(t.ids_inactive(), set())
        self.assertTrue(all(s.track_id.item() == first_tid for s in t[first_tid]))
        self.assertTrue(all(s.track_id.item() == second_tid for s in t[second_tid]))

        t.add(tracks={0: DUMMY_STATE.copy()}, new_tracks=[])
        self.assertTrue(d == MULTI_TRACKS.data[k] for k, d in t.data.items())
        self.assertEqual(len(t), 2)
        self.assertEqual(t.ids_active(), {first_tid})
        self.assertEqual(t.ids_inactive(), {second_tid})
        self.assertTrue(all(s.track_id.item() == first_tid for s in t[first_tid]))
        self.assertTrue(all(s.track_id.item() == second_tid for s in t[second_tid]))

        t.add(tracks={first_tid: DUMMY_STATE.copy()}, new_tracks=[])  # second time 1 was not found -> remove
        self.assertEqual(len(t.inactive), 0)
        self.assertEqual(len(t), 1)
        self.assertEqual(t.ids_active(), {first_tid})
        self.assertEqual(t.ids_inactive(), set())
        self.assertTrue(all(s.track_id.item() == first_tid for s in t[first_tid]))

        t.add(tracks={}, new_tracks=[])  # add nothing
        self.assertEqual(len(t.inactive), 1)
        self.assertEqual(len(t), 1)
        self.assertEqual(t.ids_active(), set())
        self.assertEqual(t.ids_inactive(), {first_tid})
        self.assertTrue(all(s.track_id.item() == first_tid for s in t[first_tid]))

    def test_get_state(self):
        t1 = ONE_TRACKS.copy()
        r1 = t1.get_states()
        self.assertTrue(isinstance(r1, State))
        self.assertEqual(len(r1), 1)
        self.assertTrue(torch.allclose(r1.track_id, torch.tensor(OT_O_ID, dtype=torch.long)))

        t2 = MULTI_TRACKS.copy()
        r2 = t2.get_states()
        self.assertTrue(isinstance(r2, State))
        self.assertEqual(len(r2), 2)
        self.assertTrue(torch.allclose(r2.track_id, torch.tensor([MT_O_ID, MT_F_ID], dtype=torch.long)))

        t0 = Tracks()
        r0 = t0.get_states()
        self.assertTrue(isinstance(r2, State))
        self.assertEqual(len(r0), 0)

    def setUp(self):
        self.assertEqual(len(ONE_TRACK), 1)
        self.assertEqual(len(FULL_TRACK), MAX_LENGTH)

        for tracks, ids, act, inact in [
            (EMPTY_TRACKS, set(), set(), set()),
            (ONE_TRACKS, {0}, {0}, set()),
            (MULTI_TRACKS, {0, 1}, {0}, {1}),
        ]:
            # ids and len
            self.assertEqual(len(tracks), len(ids))
            self.assertEqual(tracks.ids(), ids)
            # active and nof active
            self.assertEqual(tracks.nof_active(), len(act))
            self.assertEqual(tracks.ids_active(), act)
            # inactive and nof_inactive
            self.assertEqual(tracks.nof_inactive(), len(inact))
            self.assertEqual(tracks.ids_inactive(), inact)

        self.assertTrue(ONE_TRACKS.copy()[OT_O_ID] == _track_w_tid(ONE_TRACK, OT_O_ID))
        self.assertTrue(MULTI_TRACKS.copy()[MT_F_ID] == _track_w_tid(FULL_TRACK, MT_F_ID))

    def tearDown(self):
        self.assertEqual(len(ONE_TRACK), 1)
        self.assertEqual(len(FULL_TRACK), MAX_LENGTH)

        for tracks, ids, act, inact in [
            (EMPTY_TRACKS, set(), set(), set()),
            (ONE_TRACKS, {0}, {0}, set()),
            (MULTI_TRACKS, {0, 1}, {0}, {1}),
        ]:
            # ids and len
            self.assertEqual(len(tracks), len(ids))
            self.assertEqual(tracks.ids(), ids)
            # active and nof active
            self.assertEqual(tracks.nof_active(), len(act))
            self.assertEqual(tracks.ids_active(), act)
            # inactive and nof_inactive
            self.assertEqual(tracks.nof_inactive(), len(inact))
            self.assertEqual(tracks.ids_inactive(), inact)

        self.assertTrue(ONE_TRACKS.copy()[OT_O_ID] == _track_w_tid(ONE_TRACK, OT_O_ID))
        self.assertTrue(MULTI_TRACKS.copy()[MT_F_ID] == _track_w_tid(FULL_TRACK, MT_F_ID))


if __name__ == "__main__":
    unittest.main()
