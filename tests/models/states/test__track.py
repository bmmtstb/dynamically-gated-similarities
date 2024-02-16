import unittest

import torch

from dgs.models.states import Track
from helper import test_multiple_devices
from .test__queue import EMPTY_QUEUE, FULL_QUEUE, ONE_QUEUE

MAX_LENGTH: int = 30

EMPTY_TRACK: Track = Track(N=MAX_LENGTH)
ONE_TRACK: Track = Track(N=MAX_LENGTH, states={"one": ONE_QUEUE.copy()})
MULT_TRACK: Track = Track(N=MAX_LENGTH, states={str(i): ONE_QUEUE.copy() for i in range(10)})
FULL_TRACK: Track = Track(N=MAX_LENGTH, states={"one": ONE_QUEUE.copy(), "full": FULL_QUEUE.copy()})


class TestTrack(unittest.TestCase):
    def test_get_item(self):
        empty = EMPTY_TRACK.copy()
        self.assertEqual(empty[-1], {})

        one = ONE_TRACK.copy()
        self.assertEqual(one[-1], {"one": torch.ones(1, dtype=torch.int)})
        self.assertEqual(one[0], {"one": torch.ones(1, dtype=torch.int)})

        full = FULL_TRACK.copy()
        self.assertEqual(
            full[-1], {"one": torch.ones(1, dtype=torch.int), "full": (MAX_LENGTH - 1) * torch.ones(1, dtype=torch.int)}
        )

    @test_multiple_devices
    def test_append(self, device: torch.device):
        t = EMPTY_TRACK.copy().to(device=device)
        t.append({"one": torch.ones(1, dtype=torch.int, device=device)})
        self.assertEqual(t, ONE_TRACK.copy().to(device=device))
        self.assertEqual(len(t.get_queue("one")), 1)

        for i in range(MAX_LENGTH):
            t.append({"full": torch.ones(1, dtype=torch.int, device=device) * i})
            self.assertEqual(len(t.get_queue("one")), 1)
            self.assertEqual(len(t.get_queue("full")), i + 1)
        self.assertEqual(t, FULL_TRACK.copy().to(device=device))

    def test_init(self):
        with self.assertRaises(ValueError) as e:
            _ = Track(N=MAX_LENGTH + 1, states={"faulty": ONE_QUEUE.copy()})
        self.assertTrue("Provided states must have max_length" in str(e.exception), msg=e.exception)

        t = Track(N=MAX_LENGTH, states={"empty": EMPTY_QUEUE.copy()})
        self.assertEqual(len(t), 0)

    def test_len(self):
        self.assertEqual(len(EMPTY_TRACK.copy()), 0)
        self.assertEqual(len(ONE_TRACK.copy()), 1)
        self.assertEqual(len(MULT_TRACK.copy()), 1)

        # track with empty queue
        t = Track(N=MAX_LENGTH, states={"empty": EMPTY_QUEUE.copy()})
        self.assertEqual(len(t), 0)

        # queues of different length
        t = FULL_TRACK.copy()
        t.append({"dummy": torch.zeros(1)})
        with self.assertRaises(IndexError) as e:
            _ = len(t)
        self.assertTrue("Queues have different length." in str(e.exception), msg=e.exception)
        with self.assertRaises(IndexError) as e:
            _ = len(FULL_TRACK.copy())
        self.assertTrue("Queues have different length." in str(e.exception), msg=e.exception)

    def setUp(self):
        self.assertEqual(len(EMPTY_TRACK.copy()), 0)
        self.assertEqual(len(ONE_TRACK.copy()), 1)
        self.assertEqual(len(MULT_TRACK.copy()), 1)

    def tearDown(self):
        self.assertEqual(len(EMPTY_TRACK.copy()), 0)
        self.assertEqual(len(ONE_TRACK.copy()), 1)
        self.assertEqual(len(MULT_TRACK.copy()), 1)


if __name__ == "__main__":
    unittest.main()
