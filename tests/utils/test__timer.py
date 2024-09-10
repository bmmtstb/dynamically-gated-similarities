import time
import unittest

from dgs.utils.timer import DifferenceTimer, DifferenceTimers


class TestDifferenceTimer(unittest.TestCase):
    def test_add_multiple(self):
        timer = DifferenceTimer()
        N = 10
        self.assertEqual(len(timer), 0)
        for _ in range(N):
            start_time = time.time()
            timer.add(start_time)
        self.assertEqual(len(timer), N)

    def test_add_with_now(self):
        diff = 0.1
        timer = DifferenceTimer()
        start_time = time.time()
        end_time = time.time() + diff
        timer.add(prev_time=start_time, now=end_time)
        self.assertEqual(len(timer), 1)
        self.assertAlmostEqual(timer.sum(), diff, places=2)

    def test_average(self):
        for data, avg, avg_hms in [
            ([], 0.0, "0:00:00"),
            ([1.0], 1.0, "0:00:01"),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 3.0, "0:00:03"),
            ([100, 200, 300, 400, 500], 300.0, "0:05:00"),
            ([60 * 60], 3_600, "1:00:00"),
            ([60 * 60 * 10], 36_000, "10:00:00"),
        ]:
            with self.subTest(msg="data: {}, avg: {}, hms: {}".format(data, avg, avg_hms)):
                timer = DifferenceTimer()
                timer.data = data
                self.assertEqual(timer.average(), avg)
                self.assertEqual(timer.avg_hms(), avg_hms)

    def test_sum(self):
        for data, sum_, sum_hms in [
            ([], 0.0, "0:00:00"),
            ([1.0], 1.0, "0:00:01"),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 15.0, "0:00:15"),
            ([100, 200, 300, 400, 500], 1500.0, "0:25:00"),
            ([60 * 60], 3_600, "1:00:00"),
            ([60 * 60 * 10], 36_000, "10:00:00"),
        ]:
            with self.subTest(msg="data: {}, sum: {}, hms: {}".format(data, sum_, sum_hms)):
                timer = DifferenceTimer()
                timer.data = data
                self.assertEqual(timer.sum(), sum_)
                self.assertEqual(timer.sum_hms(), sum_hms)


class TestDifferenceTimers(unittest.TestCase):
    def setUp(self):
        self.empty_timers = DifferenceTimers()

        self.two_timers = DifferenceTimers()
        self.two_timers.add(name="timer1", prev_time=time.time())
        self.two_timers.add(name="timer2", prev_time=time.time())

        self.diff: float = 0.1

        self.assertEqual(len(self.empty_timers), 0)
        self.assertFalse("timer1" in self.empty_timers)

    def test_add(self):
        prev_time = time.time() - self.diff
        diff = self.empty_timers.add(name="timer1", prev_time=prev_time)
        self.assertTrue(isinstance(self.empty_timers["timer1"], DifferenceTimer))
        self.assertAlmostEqual(diff, self.diff, places=2)

    def test_add_with_now(self):
        prev_time = time.time()
        now = time.time() + self.diff
        diff = self.empty_timers.add(name="timer1", prev_time=prev_time, now=now)
        self.assertTrue(isinstance(self.empty_timers["timer1"], DifferenceTimer))
        self.assertAlmostEqual(diff, self.diff, places=2)

    def test_add_multiple(self):
        old_times = {"timer1": time.time() - self.diff, "timer2": time.time() - self.diff}
        diffs = self.two_timers.add_multiple(prev_times=old_times)
        self.assertTrue(isinstance(self.two_timers["timer1"], DifferenceTimer))
        self.assertTrue(isinstance(self.two_timers["timer2"], DifferenceTimer))
        self.assertAlmostEqual(diffs["timer1"], self.diff, places=2)
        self.assertAlmostEqual(diffs["timer2"], self.diff, places=2)

    def test_add_multiple_new_timers(self):
        prev_times = {"new_timer1": time.time() - self.diff, "new_timer2": time.time() - self.diff}
        diffs = self.empty_timers.add_multiple(prev_times=prev_times)
        self.assertTrue(isinstance(self.empty_timers["new_timer1"], DifferenceTimer))
        self.assertTrue(isinstance(self.empty_timers["new_timer2"], DifferenceTimer))
        self.assertAlmostEqual(diffs["new_timer1"], self.diff, places=2)
        self.assertAlmostEqual(diffs["new_timer2"], self.diff, places=2)

    def test_get_sums(self):
        old_times = {"timer1": time.time() - self.diff, "timer2": time.time() - self.diff}
        self.empty_timers.add_multiple(prev_times=old_times)
        sums = self.empty_timers.get_sums()
        self.assertAlmostEqual(sums["timer1"], self.diff, places=2)
        self.assertAlmostEqual(sums["timer2"], self.diff, places=2)
        # add a second time
        old_times_2 = {"timer1": time.time() - self.diff, "timer2": time.time() - self.diff}
        self.empty_timers.add_multiple(prev_times=old_times_2)
        sums = self.empty_timers.get_sums()
        self.assertAlmostEqual(sums["timer1"], 2 * self.diff, places=2)
        self.assertAlmostEqual(sums["timer2"], 2 * self.diff, places=2)

    def test_get_avgs(self):
        old_times = {"timer1": time.time() - self.diff, "timer2": time.time() - self.diff}
        self.empty_timers.add_multiple(prev_times=old_times)
        avgs = self.empty_timers.get_avgs()
        self.assertAlmostEqual(avgs["timer1"], self.diff, places=2)
        self.assertAlmostEqual(avgs["timer2"], self.diff, places=2)
        # add a second time
        old_times_2 = {"timer1": time.time() - self.diff, "timer2": time.time() - self.diff}
        self.empty_timers.add_multiple(prev_times=old_times_2)
        avgs = self.empty_timers.get_avgs()
        self.assertAlmostEqual(avgs["timer1"], self.diff, places=2)
        self.assertAlmostEqual(avgs["timer2"], self.diff, places=2)

    def test_add_new_timer(self):
        prev_time = time.time() - self.diff
        diff = self.empty_timers.add(name="new_timer", prev_time=prev_time)
        self.assertAlmostEqual(diff, self.diff, places=2)

    def test_get_sums_empty(self):
        sums = self.empty_timers.get_sums()
        self.assertEqual(sums, {})

    def test_get_avgs_empty(self):
        avgs = self.empty_timers.get_avgs()
        self.assertEqual(avgs, {})

    def test_get_last(self):
        prev_time = time.time() - self.diff
        self.empty_timers.add(name="timer1", prev_time=prev_time)
        last = self.empty_timers.get_last()
        self.assertAlmostEqual(last["timer1"], self.diff, places=2)


if __name__ == "__main__":
    unittest.main()
