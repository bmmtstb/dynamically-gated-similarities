import unittest

import torch

from dgs.utils.states import Track
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

        with self.assertRaises(ValueError) as e:
            t.append({})
        self.assertTrue("Can not append an empty state" in str(e.exception), msg=e.exception)

    def test_init(self):
        with self.assertRaises(ValueError) as e:
            _ = Track(N=MAX_LENGTH + 1, states={"faulty": ONE_QUEUE.copy()})
        self.assertTrue("Provided states must have max_length" in str(e.exception), msg=e.exception)

        t = Track(N=MAX_LENGTH, states={"empty": EMPTY_QUEUE.copy()})
        self.assertEqual(len(t), 0)

        with self.assertRaises(ValueError) as e:
            _ = Track(N=-1)
        self.assertTrue("N must be greater than 0 but got" in str(e.exception), msg=e.exception)

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

    def test_equality(self):
        self.assertTrue(EMPTY_TRACK == EMPTY_TRACK)
        self.assertTrue(ONE_TRACK == ONE_TRACK)
        self.assertTrue(FULL_TRACK == FULL_TRACK)

        self.assertFalse(EMPTY_TRACK == ONE_TRACK)
        self.assertFalse(EMPTY_TRACK == FULL_TRACK)
        self.assertFalse(ONE_TRACK == FULL_TRACK)
        self.assertFalse(ONE_TRACK == {})

    def test_get_states(self):
        self.assertEqual(EMPTY_TRACK.get_states(), {})
        self.assertEqual(ONE_TRACK.get_states(), {"one": ONE_QUEUE.copy()})
        self.assertEqual(FULL_TRACK.get_states(), {"one": ONE_QUEUE.copy(), "full": FULL_QUEUE.copy()})

    def test_get_state(self):
        self.assertEqual(ONE_TRACK.get_state(-1), {"one": ONE_QUEUE.copy()[-1]})
        self.assertEqual(FULL_TRACK.get_state(0), {"one": ONE_QUEUE.copy()[0], "full": FULL_QUEUE.copy()[0]})

        with self.assertRaises(IndexError) as e:
            _ = EMPTY_TRACK.get_state(MAX_LENGTH)
        self.assertTrue("is larger than the maximum number of items in a queue" in str(e.exception), msg=e.exception)

        with self.assertRaises(IndexError) as e:
            _ = EMPTY_TRACK.get_state(-(MAX_LENGTH + 1))
        self.assertTrue("is smaller than the maximum number of items in a queue" in str(e.exception), msg=e.exception)

        with self.assertRaises(IndexError) as e:
            _ = FULL_TRACK.get_state(1)
        self.assertTrue("is out of bounds for at least one query" in str(e.exception), msg=e.exception)

        with self.assertRaises(IndexError) as e:
            _ = FULL_TRACK.get_state(-2)
        self.assertTrue("is out of bounds for at least one query" in str(e.exception), msg=e.exception)

    def test_get_queue(self):
        self.assertEqual(ONE_TRACK.copy().get_queue("one"), ONE_QUEUE)

        with self.assertRaises(KeyError) as e:
            _ = ONE_TRACK.copy().get_queue("faulty")
        self.assertTrue("does not exist in the current states" in str(e.exception), msg=e.exception)

        with self.assertRaises(KeyError) as e:
            _ = EMPTY_TRACK.copy().get_queue("")
        self.assertTrue("does not exist in the current states" in str(e.exception), msg=e.exception)

        self.assertEqual(FULL_TRACK.copy().get_queue("one"), ONE_QUEUE)
        self.assertEqual(FULL_TRACK.copy().get_queue("full"), FULL_QUEUE)

    def test_get_queues(self):
        self.assertEqual(EMPTY_TRACK.copy().get_queues([]), {})

        with self.assertRaises(IndexError) as e:
            _ = ONE_TRACK.copy().get_queues(["faulty"])
        self.assertTrue("One of the provided names does not exist" in str(e.exception), msg=e.exception)

        self.assertEqual(FULL_TRACK.copy().get_queues(["one", "full"]), FULL_TRACK.copy().get_states())

    def test_prop_names(self):
        self.assertEqual(EMPTY_TRACK.names, [])
        self.assertEqual(ONE_TRACK.names, ["one"])
        self.assertEqual(FULL_TRACK.names, ["one", "full"])

    def test_prop_N(self):
        self.assertEqual(EMPTY_TRACK.N, MAX_LENGTH)
        self.assertEqual(FULL_TRACK.N, MAX_LENGTH)
        self.assertEqual(Track(N=1).N, 1)

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
