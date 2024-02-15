import unittest

import torch

from dgs.models.states import Track
from helper import test_multiple_devices
from .test__queue import FULL_QUEUE, ONE_QUEUE

MAX_LENGTH: int = 30

EMPTY_TRACK: Track = Track(N=MAX_LENGTH)
ONE_TRACK: Track = Track(N=MAX_LENGTH, states={"one": ONE_QUEUE})
FULL_TRACK: Track = Track(N=MAX_LENGTH, states={"one": ONE_QUEUE, "full": FULL_QUEUE})


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


if __name__ == "__main__":
    unittest.main()
