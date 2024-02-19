import unittest

import torch

from dgs.models.states import Queue
from helper import test_multiple_devices

MAX_LENGTH: int = 30

EMPTY_QUEUE: Queue = Queue(N=MAX_LENGTH, shape=torch.Size((1,)))
ONE_QUEUE: Queue = Queue(N=MAX_LENGTH, states=[torch.ones(1, dtype=torch.int)])
FULL_QUEUE: Queue = Queue(N=MAX_LENGTH, states=[torch.ones(1, dtype=torch.int) * i for i in range(MAX_LENGTH)])


class TestQueue(unittest.TestCase):

    def test_init(self):
        q = EMPTY_QUEUE.copy()
        self.assertEqual(len(q), 0)
        self.assertEqual(q.N, MAX_LENGTH)

        q = ONE_QUEUE.copy()
        self.assertEqual(len(q), 1)
        self.assertEqual(q.N, MAX_LENGTH)

        with self.assertRaises(ValueError) as e:
            _ = Queue(N=MAX_LENGTH, shape=torch.Size((2,)), states=[torch.ones(1)])
        self.assertTrue("First shape of the values in states" in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = Queue(N=-1)
        self.assertTrue("N must be greater than 0 but got" in str(e.exception), msg=e.exception)

    def test_get_item(self):
        empty_q = EMPTY_QUEUE.copy()
        with self.assertRaises(IndexError):
            _ = empty_q[0]

        q = Queue(N=MAX_LENGTH, states=[torch.ones(1, dtype=torch.int) * i for i in range(MAX_LENGTH)])
        for i in range(MAX_LENGTH):
            self.assertEqual(q[i].item(), i)

    def test_append(self):
        q = Queue(N=MAX_LENGTH)  # without shape !
        for i in range(MAX_LENGTH + 3):
            q.append(torch.ones(1) * i)
            self.assertEqual(len(q), min(i + 1, MAX_LENGTH))
            self.assertTrue(torch.allclose(q[-1], torch.ones(1) * i))

    def test_append_wrong_shape(self):
        q = ONE_QUEUE.copy()
        with self.assertRaises(ValueError) as e:
            q.append(torch.ones(2))
        self.assertTrue("The shape of the new state" in str(e.exception), msg=e.exception)

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

    def test_get_device_on_empty(self):
        q = EMPTY_QUEUE.copy()
        with self.assertRaises(ValueError) as e:
            _ = q.device
        self.assertTrue("Can not get the device of an empty Queue" in str(e.exception), msg=e.exception)

    def test_shape(self):
        empty_q = Queue(N=MAX_LENGTH)
        with self.assertRaises(ValueError) as e:
            _ = empty_q.shape
        self.assertTrue("Can not get the shape of an empty Queue." in str(e.exception), msg=e.exception)

        empty_q = Queue(N=MAX_LENGTH, shape=torch.Size((1, 2, 3)))
        self.assertEqual(empty_q.shape, torch.Size((1, 2, 3)))

    def setUp(self):
        self.assertEqual(len(EMPTY_QUEUE.copy()), 0)
        self.assertEqual(len(ONE_QUEUE.copy()), 1)
        self.assertEqual(len(FULL_QUEUE.copy()), MAX_LENGTH)

    def tearDown(self):
        self.assertEqual(len(EMPTY_QUEUE.copy()), 0)
        self.assertEqual(len(ONE_QUEUE.copy()), 1)
        self.assertEqual(len(FULL_QUEUE.copy()), MAX_LENGTH)


if __name__ == "__main__":
    unittest.main()
