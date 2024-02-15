import unittest

import torch

from dgs.models.states import Queue
from helper import test_multiple_devices

MAX_LENGTH: int = 30

EMPTY_QUEUE: Queue = Queue(N=MAX_LENGTH, shape=torch.Size((1,)))
ONE_QUEUE: Queue = Queue(N=MAX_LENGTH, states=[torch.ones(1, dtype=torch.int)])
FULL_QUEUE: Queue = Queue(N=MAX_LENGTH, states=[torch.ones(1, dtype=torch.int) * i for i in range(MAX_LENGTH)])


class TestQueue(unittest.TestCase):

    def test_get_item(self):
        empty_q = EMPTY_QUEUE.copy()
        with self.assertRaises(IndexError):
            _ = empty_q[0]

        q = Queue(N=MAX_LENGTH, states=[torch.ones(1, dtype=torch.int) * i for i in range(MAX_LENGTH)])
        for i in range(MAX_LENGTH):
            self.assertEqual(q[i].item(), i)

    def test_append(self):
        q = Queue(N=MAX_LENGTH)
        for i in range(MAX_LENGTH + 3):
            q.append(torch.ones(1) * i)
            self.assertEqual(len(q), min(i + 1, MAX_LENGTH))
            self.assertTrue(torch.allclose(q[-1], torch.ones(1) * i))

    def test_init(self):
        q = Queue(N=MAX_LENGTH)
        self.assertEqual(len(q), 0)

        q = Queue(N=MAX_LENGTH, states=[torch.ones(1)])
        self.assertEqual(len(q), 1)

        with self.assertRaises(ValueError) as e:
            _ = Queue(N=MAX_LENGTH, shape=torch.Size((2,)), states=[torch.ones(1)])
        self.assertTrue("First shape of the values in states" in str(e.exception), msg=e.exception)

    def test_clear(self):
        empty = EMPTY_QUEUE.copy()
        empty.clear()
        self.assertEqual(empty, EMPTY_QUEUE)

        full = FULL_QUEUE.copy()
        self.assertEqual(len(full), MAX_LENGTH)
        full.clear()
        self.assertEqual(len(full), 0)
        self.assertEqual(full, EMPTY_QUEUE)

    @test_multiple_devices
    def test_to(self, device):
        q = Queue(N=MAX_LENGTH, states=[torch.ones(1, device="cpu")])
        self.assertEqual(q.device.type, "cpu")
        q.to(device=device)
        self.assertEqual(q[0].device, device)
        self.assertEqual(q.device, device)

    @test_multiple_devices
    def test_get_all(self, device):
        empty_q = Queue(N=MAX_LENGTH)
        with self.assertRaises(ValueError) as e:
            _ = empty_q.get_all()
        self.assertTrue("Can not stack the items of an empty Queue." in str(e.exception), msg=e.exception)

        q = Queue(N=MAX_LENGTH, states=[torch.ones(1, device=device, dtype=torch.int) * i for i in range(MAX_LENGTH)])
        stacked = q.get_all()
        self.assertTrue(
            torch.allclose(stacked, torch.arange(start=0, end=MAX_LENGTH, dtype=torch.int, device=device).view((-1, 1)))
        )
        self.assertTrue(stacked.device, device)

    def test_shape(self):
        empty_q = Queue(N=MAX_LENGTH)
        with self.assertRaises(ValueError) as e:
            _ = empty_q.shape
        self.assertTrue("Can not get the shape of an empty Queue." in str(e.exception), msg=e.exception)

        empty_q = Queue(N=MAX_LENGTH, shape=torch.Size((1, 2, 3)))
        self.assertEqual(empty_q.shape, torch.Size((1, 2, 3)))


if __name__ == "__main__":
    unittest.main()
