import unittest

from dgs.utils.track import TrackStatistics


class TestTrackStatistics(unittest.TestCase):

    def test_init(self):
        ts = TrackStatistics()
        self.assertEqual(ts.active, set())
        self.assertEqual(ts.inactive, set())
        self.assertEqual(ts.removed, [])

    def test_add(self):
        ts = TrackStatistics()

        ts.new.append(1)
        ts.reactivated.append(2)
        ts.found.append(3)
        ts.still_active.append(4)
        self.assertEqual(ts.nof_active, 4)
        self.assertEqual(ts.nof_new, 1)
        self.assertEqual(ts.nof_reactivated, 1)
        self.assertEqual(ts.nof_found, 1)
        self.assertEqual(ts.nof_still_active, 1)

        ts.removed.append(5)
        self.assertEqual(ts.nof_removed, 1)

        ts.lost.append(6)
        ts.still_inactive.append(1)
        self.assertEqual(ts.nof_inactive, 2)
        self.assertEqual(ts.nof_lost, 1)
        self.assertEqual(ts.nof_still_inactive, 1)

        ts.clear()

        self.assertEqual(ts.active, set())
        self.assertEqual(ts.inactive, set())
        self.assertEqual(ts.removed, [])

    def test_add_duplicates(self):
        ts = TrackStatistics()

        ts.new.append(1)
        ts.found.append(1)
        ts.still_active.append(1)

        self.assertEqual(ts.active, {1})


if __name__ == "__main__":
    unittest.main()
