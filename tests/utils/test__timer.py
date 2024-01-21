import time
import unittest

from dgs.utils.timer import DifferenceTimer


class TestDifferenceTimer(unittest.TestCase):
    def test_add_multiple(self):
        timer = DifferenceTimer()
        N = 10
        self.assertEqual(len(timer), 0)
        for _ in range(N):
            start_time = time.time()
            timer.add(start_time)
        self.assertEqual(len(timer), N)

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


if __name__ == "__main__":
    unittest.main()
