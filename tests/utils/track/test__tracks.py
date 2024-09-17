import unittest

import torch as t
from torchvision.tv_tensors import BoundingBoxes

from dgs.utils.config import DEF_VAL
from dgs.utils.state import EMPTY_STATE, State
from dgs.utils.track import Track, Tracks, TrackStatus
from helper import test_multiple_devices


def _track_w_params(
    track: Track, tid: int = 0, nof_a: int = 0, status: TrackStatus = TrackStatus.New, start: int = 0
) -> Track:
    track_ = track.copy()
    track_.id = tid
    track_._nof_active = nof_a
    track_._status = status
    track_._start_frame = start
    return track_


THRESH: int = 2

# TRACK
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


# TRACKS
EMPTY_TRACKS: Tracks = Tracks(N=MAX_LENGTH, thresh=THRESH)

ONE_TRACKS: Tracks = Tracks(N=MAX_LENGTH, thresh=THRESH)
OT_O_ID = ONE_TRACKS.add(tracks={}, new=[DUMMY_STATE.copy()])[0]

MULTI_TRACKS: Tracks = Tracks(N=MAX_LENGTH, thresh=THRESH)
MT_ACT_ID = MULTI_TRACKS.add(tracks={}, new=[DUMMY_STATE.copy()])[0]
MT_DEL_ID = MULTI_TRACKS.add(tracks={MT_ACT_ID: DUMMY_STATE.copy()}, new=[DUMMY_STATE])[0]
for _ in range(MAX_LENGTH - 1):
    MULTI_TRACKS[MT_DEL_ID].append(DUMMY_STATE)
MT_INA_IDS = MULTI_TRACKS.add(tracks={MT_ACT_ID: DUMMY_STATE.copy()}, new=DUMMY_STATES.copy())
MULTI_TRACKS.add(tracks={MT_ACT_ID: DUMMY_STATE.copy()}, new=[])
# now MT_F_IDS is removed and MT_M_IDS are all inactive !


class TestTracks(unittest.TestCase):

    def _test_constants(self):
        self.assertEqual(len(ONE_TRACK), 1)
        self.assertEqual(len(FULL_TRACK), MAX_LENGTH)

        tracks: Tracks

        for tracks, ids, act, inact, removed, age, ages, stati, nof_acts in [
            (EMPTY_TRACKS, set(), set(), set(), set(), 0, {}, {}, {}),
            (ONE_TRACKS, {0}, {0}, set(), set(), 1, {OT_O_ID: 1}, {OT_O_ID: TrackStatus.Active}, {OT_O_ID: 1}),
            (
                MULTI_TRACKS,
                {i for i in range(MAX_LENGTH + 2) if i != 1},
                {0},
                {i for i in range(2, MAX_LENGTH + 2)},
                {MT_DEL_ID},
                3,
                {MT_ACT_ID: 4, MT_DEL_ID: 0, **{tid: 1 for tid in MT_INA_IDS}},
                {
                    MT_ACT_ID: TrackStatus.Active,
                    MT_DEL_ID: TrackStatus.Removed,
                    **{tid: TrackStatus.Inactive for tid in MT_INA_IDS},
                },
                {MT_ACT_ID: 4, MT_DEL_ID: 0, **{tid: 0 for tid in MT_INA_IDS}},
            ),
        ]:
            # ids and len
            self.assertEqual(len(tracks), len(ids))
            self.assertEqual(tracks.ids, ids)
            for tid in ids:
                self.assertEqual(tracks[tid].id, tid)
            # active and nof active
            self.assertEqual(tracks.nof_active, len(act))
            self.assertEqual(tracks.ids_active, act)
            # inactive and nof_inactive
            self.assertEqual(tracks.nof_inactive, len(inact))
            self.assertEqual(tracks.ids_inactive, inact)
            # removed
            self.assertEqual(set(tracks.ids_removed), removed)
            # status
            for tid, status in stati.items():
                if tid in tracks.data:
                    self.assertEqual(tracks[tid].status, status)
                else:
                    self.assertEqual(tracks.removed[tid].status, status)
            # nof_active
            for tid, nof_act in nof_acts.items():
                if tid in tracks.data:
                    self.assertEqual(tracks[tid].nof_active, nof_act)
                else:
                    self.assertEqual(tracks.removed[tid].nof_active, nof_act)

    def test_init(self):
        tr = Tracks(N=MAX_LENGTH, thresh=THRESH)
        self.assertEqual(len(tr.data), 0)
        self.assertEqual(len(tr.inactive), 0)
        self.assertEqual(tr.inactivity_threshold, THRESH)

        # test without explicit thresh
        no_thresh = Tracks(N=MAX_LENGTH)
        self.assertEqual(len(no_thresh.data), 0)
        self.assertEqual(len(no_thresh.inactive), 0)
        self.assertEqual(no_thresh.inactivity_threshold, DEF_VAL["tracks"]["inactivity_threshold"])

        with self.assertRaises(ValueError) as e:
            _ = Tracks(N=-1)
        self.assertTrue("N must be greater than 0 but got '-1'" in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = Tracks(N=MAX_LENGTH, thresh=-1)
        self.assertTrue("Threshold must be positive, got -1." in str(e.exception), msg=e.exception)

        with self.assertRaises(TypeError) as e:
            # noinspection PyTypeChecker
            _ = Tracks(N=MAX_LENGTH, thresh="1")
        self.assertTrue("Threshold is expected to be int or None, but got " in str(e.exception), msg=e.exception)

    def test_equality(self):
        self.assertTrue(EMPTY_TRACKS == EMPTY_TRACKS)
        self.assertTrue(ONE_TRACKS == ONE_TRACKS)

        self.assertFalse("dummy" == EMPTY_TRACKS)
        self.assertFalse(EMPTY_TRACKS == Tracks(N=MAX_LENGTH, thresh=THRESH - 1))

    def test_copy(self):
        t1 = ONE_TRACKS.copy()
        tr = t1.copy()
        tr.reset()
        t1.add(tracks={OT_O_ID: DUMMY_STATE.copy()}, new=[DUMMY_STATE.copy()])
        self.assertEqual(len(tr), 0)
        self.assertEqual(len(t1), 2)

    def test_reset(self):
        for tracks in [EMPTY_TRACKS, ONE_TRACKS, MULTI_TRACKS]:
            with self.subTest(msg="tracks: {}".format(tracks)):
                tr = tracks.copy()
                tr.reset()
                self.assertTrue(tr == EMPTY_TRACKS)

        self.assertEqual(len(ONE_TRACKS[OT_O_ID]), 1)
        self.assertEqual(len(MULTI_TRACKS[MT_ACT_ID]), 4)
        self.assertEqual(len(MULTI_TRACKS.removed[MT_DEL_ID]), 30)
        for tid in MT_INA_IDS:
            self.assertEqual(len(MULTI_TRACKS[tid]), 1)

    def test_next_frame(self):
        tr = EMPTY_TRACKS.copy()
        self.assertEqual(tr._curr_frame, 0)
        tr._next_frame()
        self.assertEqual(tr._curr_frame, 1)

    def test_remove_tid(self):
        t0 = EMPTY_TRACKS.copy()
        with self.assertRaises(KeyError) as e:
            t0.remove_tid(0)
        self.assertTrue("Track-ID 0 can not be deleted, because it is not present" in str(e.exception), msg=e.exception)
        self.assertEqual(len(t0), 0)
        self.assertTrue(t0 == EMPTY_TRACKS)

        t1 = ONE_TRACKS.copy()
        track1 = ONE_TRACKS[OT_O_ID].copy()
        self.assertEqual(len(t1), 1)
        t1.remove_tid(OT_O_ID)
        self.assertEqual(len(t1), 0)
        self.assertTrue(t1 == EMPTY_TRACKS)
        self.assertTrue(isinstance(t1.removed[OT_O_ID], Track))
        self.assertEqual(t1.removed[OT_O_ID], _track_w_params(track1, tid=-1, status=TrackStatus.Removed))

        t2 = MULTI_TRACKS.copy()
        self.assertEqual(len(t2), 1 + MAX_LENGTH)
        t2.remove_tids(MT_INA_IDS)
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2.ids, {MT_ACT_ID})
        self.assertEqual(t2.ids_active, {MT_ACT_ID})
        self.assertEqual(t2.ids_inactive, set())
        self.assertEqual(t2.ids_removed, set([MT_DEL_ID] + MT_INA_IDS))

        ts_own = Tracks(N=MAX_LENGTH, thresh=THRESH)
        own_tid = ts_own.add({}, [DUMMY_STATE.copy()])[0]
        t_own = ts_own[own_tid]
        self.assertEqual(t_own.status, TrackStatus.Active)
        ts_own.remove_tid(own_tid)
        custom_empty = EMPTY_TRACKS.copy()
        custom_empty._next_frame()
        self.assertTrue(ts_own == custom_empty)
        self.assertEqual(t_own.status, TrackStatus.Removed)

    def test_reactivate_track(self):
        tr = MULTI_TRACKS.copy()
        tr.reactivate_track(MT_DEL_ID)
        self.assertTrue(MT_DEL_ID in tr.ids)
        self.assertFalse(MT_DEL_ID in tr.ids_removed)
        self.assertFalse(MT_DEL_ID in tr.removed)
        self.assertEqual(tr[MT_DEL_ID].status, TrackStatus.Reactivated)
        self.assertEqual(tr[MT_DEL_ID].nof_active, 0)
        self.assertEqual(tr[MT_DEL_ID]._start_frame, 1)

        with self.assertRaises(KeyError) as e:
            EMPTY_TRACKS.copy().reactivate_track(1)
        self.assertTrue("Track-ID 1 not present in removed Tracks." in str(e.exception), msg=e.exception)

    def test_get_next_id(self):
        t0 = EMPTY_TRACKS.copy()
        self.assertEqual(t0._get_next_id(), 0)

        t1 = ONE_TRACKS.copy()
        self.assertEqual(t1._get_next_id(), 1)

    @test_multiple_devices
    def test_to(self, device):
        tr = ONE_TRACKS.copy()
        self.assertTrue(tr[OT_O_ID].device == t.device("cpu"))
        tr.to(device=device)
        self.assertTrue(tr[OT_O_ID].device == device)

    def test_handle_inactive(self):
        tr = Tracks(N=MAX_LENGTH, thresh=2)
        tid = tr.add({}, new=[DUMMY_STATE.copy()])[0]
        self.assertEqual(tr.ids_active, {tid})
        self.assertEqual(tr.ids_inactive, set())

        tr._handle_inactive({tid})
        self.assertEqual(tr.ids_active, set())
        self.assertEqual(tr.ids_inactive, {tid})
        self.assertEqual(tr.inactive[tid], 1)

        tr._handle_inactive({tid})
        self.assertEqual(tr.ids_active, set())
        self.assertEqual(tr.ids_inactive, set())

    def test_add_empty_tracks(self):
        t0 = EMPTY_TRACKS.copy()

        tid0 = t0.add_empty_tracks(0)
        self.assertEqual(tid0, [])

        tid1 = t0.add_empty_tracks()
        self.assertEqual(tid1, [0])
        tid1 = tid1[0]
        self.assertTrue(t0.data[tid1] == _track_w_params(EMPTY_TRACK, tid=tid1))

        tid2 = t0.add_empty_tracks(2)
        self.assertEqual(tid2, [1, 2])
        for tid in tid2:
            self.assertTrue(t0.data[tid] == _track_w_params(EMPTY_TRACK, tid=tid))

    def test_update_track(self):
        tracks = ONE_TRACKS.copy()

        self.assertEqual(len(tracks[OT_O_ID]), 1)
        tracks._update_track(OT_O_ID, DUMMY_STATE)
        self.assertEqual(len(tracks[OT_O_ID]), 2)

        with self.assertRaises(KeyError) as e:
            tracks._update_track(100, DUMMY_STATE)
        self.assertTrue(
            "Track-ID 100 neither present in the current or previously removed Tracks" in str(e.exception),
            msg=e.exception,
        )

        # original
        multi_tracks = MULTI_TRACKS.copy()
        self.assertEqual(multi_tracks.ids_inactive, set(MT_INA_IDS))
        for tid in MT_INA_IDS:
            self.assertEqual(len(multi_tracks[tid]), 1)
        self.assertEqual(multi_tracks.ids_active, {MT_ACT_ID})
        self.assertEqual(multi_tracks.ids_removed, {MT_DEL_ID})

        # update inactive
        for tid in MT_INA_IDS:
            multi_tracks._update_track(tid, DUMMY_STATE)
            self.assertEqual(len(multi_tracks[tid]), 2)
            self.assertEqual(multi_tracks[tid][-1]["pred_tid"], tid)
        self.assertEqual(multi_tracks.ids_inactive, set())
        self.assertEqual(multi_tracks.ids_removed, {MT_DEL_ID})
        self.assertEqual(multi_tracks.ids_active, set([MT_ACT_ID] + MT_INA_IDS))

        # update removed
        multi_tracks._update_track(MT_DEL_ID, DUMMY_STATE)
        self.assertEqual(multi_tracks.ids_inactive, set())
        self.assertEqual(multi_tracks.ids_removed, set())
        self.assertEqual(multi_tracks.ids_active, set([MT_ACT_ID, MT_DEL_ID] + MT_INA_IDS))
        self.assertEqual(len(multi_tracks[MT_DEL_ID]), MAX_LENGTH)
        self.assertEqual(multi_tracks.nof_removed, 0)
        self.assertEqual(multi_tracks[MT_DEL_ID][-1]["pred_tid"], MT_DEL_ID)

    def test_get_item(self):
        empty = EMPTY_TRACKS.copy()
        with self.assertRaises(KeyError) as e:
            _ = empty[0]
        self.assertTrue("0" in str(e.exception), msg=e.exception)

        one = ONE_TRACKS.copy()
        track = one[OT_O_ID]
        self.assertTrue(isinstance(track, Track))
        self.assertTrue(track[-1], DUMMY_STATE)

    def test_reset_deleted(self):
        tr = MULTI_TRACKS.copy()
        self.assertEqual(tr.nof_removed, 1)
        tr.reset_deleted()
        self.assertEqual(tr.removed, {})
        self.assertEqual(tr.ids_removed, set())
        self.assertEqual(tr.nof_removed, 0)

    def test_add(self):
        tr = Tracks(N=MAX_LENGTH, thresh=2)

        first_tid = tr.add(tracks={}, new=[DUMMY_STATE.copy()])[0]
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr.ids_active, {first_tid})
        self.assertEqual(tr.ids_inactive, set())
        self.assertTrue(tr[first_tid].id == first_tid)

        second_tid = tr.add(tracks={0: DUMMY_STATE.copy()}, new=[DUMMY_STATE.copy()])[0]
        self.assertEqual(len(tr), 2)
        self.assertEqual(tr.ids_active, {first_tid, second_tid})
        self.assertEqual(tr.ids_inactive, set())
        self.assertTrue(tr[first_tid].id == first_tid)
        self.assertTrue(tr[second_tid].id == second_tid)

        tr.add(tracks={0: DUMMY_STATE.copy()}, new=[])
        self.assertTrue(d == MULTI_TRACKS.data[k] for k, d in tr.data.items())
        self.assertEqual(len(tr), 2)
        self.assertEqual(tr.ids_active, {first_tid})
        self.assertEqual(tr.ids_inactive, {second_tid})
        self.assertTrue(tr[first_tid].id == first_tid)
        self.assertTrue(tr[second_tid].id == second_tid)

        tr.add(tracks={first_tid: DUMMY_STATE.copy()}, new=[])  # second time 1 was not found -> remove
        self.assertEqual(len(tr.inactive), 0)
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr.ids_active, {first_tid})
        self.assertEqual(tr.ids_inactive, set())
        self.assertTrue(tr[first_tid].id == first_tid)

        tr.add(tracks={}, new=[])  # add nothing
        self.assertEqual(len(tr.inactive), 1)
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr.ids_active, set())
        self.assertEqual(tr.ids_inactive, {first_tid})
        self.assertTrue(tr[first_tid].id == first_tid)

    def test_add_empty_states(self):
        tr = Tracks(N=MAX_LENGTH, thresh=3)
        e_s: State = EMPTY_STATE.copy()

        tid = tr.add(tracks={}, new=[e_s])[0]
        self.assertEqual(tr.nof_active, 0)
        self.assertEqual(tr.nof_inactive, 1)
        _ = tr.add(tracks={tid: e_s}, new=[])
        self.assertEqual(tr.nof_active, 0)
        self.assertEqual(tr.nof_inactive, 1)
        _ = tr.add(tracks={tid: e_s}, new=[])
        self.assertEqual(tr.nof_active, 0)
        self.assertEqual(tr.nof_inactive, 0)

    def test_add_keep_inactive(self):
        tr = Tracks(N=MAX_LENGTH, thresh=5)
        tid = tr.add(tracks={}, new=[DUMMY_STATE.copy()])[0]

        tr.add(tracks={}, new=[])
        self.assertEqual(tr.ids_inactive, {tid})

        tr.add(tracks={}, new=[])
        self.assertEqual(tr.ids_inactive, {tid})

        tr.add(tracks={}, new=[])
        self.assertEqual(tr.ids_inactive, {tid})

    def test_get_states(self):
        for t, r_tids, length in [
            (Tracks(N=MAX_LENGTH), [], 0),
            (ONE_TRACKS.copy(), [OT_O_ID], 1),
            (MULTI_TRACKS.copy(), list(MULTI_TRACKS.ids), MAX_LENGTH + 1),
        ]:
            with self.subTest(msg="r_tids: {}, length: {}".format(r_tids, length)):
                s, tids = t.get_states()
                self.assertTrue(isinstance(s, list))
                self.assertTrue(all(isinstance(r, State) for r in s))
                self.assertEqual(len(s), length)
                self.assertEqual(tids, r_tids)

    def test_get_active_states(self):
        t1 = ONE_TRACKS.copy()
        r1 = t1.get_active_states()
        self.assertTrue(isinstance(r1, list))
        self.assertTrue(all(isinstance(r, State) for r in r1))
        self.assertEqual(len(r1), 1)

        t2 = MULTI_TRACKS.copy()
        r2 = t2.get_active_states()
        self.assertTrue(isinstance(r2, list))
        self.assertTrue(all(isinstance(r, State) for r in r2))
        self.assertEqual(len(r2), 1)

        t0 = Tracks(N=MAX_LENGTH)
        r0 = t0.get_active_states()
        self.assertTrue(isinstance(r0, list))
        self.assertTrue(all(isinstance(r, State) for r in r0))
        self.assertEqual(len(r0), 0)

    def test_age(self):
        age = 0
        tr = EMPTY_TRACKS.copy()
        self.assertEqual(tr.age, age)

        tr.add({}, [])
        age += 1
        self.assertEqual(tr.age, age)

        tr.reset()
        self.assertEqual(tr.age, 0)

    def test_ages(self):
        self.assertEqual(MULTI_TRACKS.ages, {MT_ACT_ID: 4, **{tid: 2 for tid in MT_INA_IDS}})

    def test_is_active_inactive_removed(self):
        tr = MULTI_TRACKS.copy()
        for tid, act, ina, rem in [
            (MT_ACT_ID, True, False, False),
            (MT_DEL_ID, False, False, True),
            *[(sub_id, False, True, False) for sub_id in MT_INA_IDS],
        ]:
            with self.subTest(msg="act: {}, ina: {}, rem: {}".format(act, ina, rem)):
                self.assertEqual(tr.is_active(tid), act)
                self.assertEqual(tr.is_inactive(tid), ina)
                self.assertEqual(tr.is_removed(tid), rem)

    def setUp(self):
        self._test_constants()

    def tearDown(self):
        self._test_constants()


if __name__ == "__main__":
    unittest.main()
