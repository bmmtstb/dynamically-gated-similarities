import unittest
from unittest.mock import patch

from torch.optim.lr_scheduler import LinearLR

from dgs.models.scheduler import get_scheduler, register_scheduler, SCHEDULERS


class Test(unittest.TestCase):

    def test_register_scheduler(self):
        with patch.dict(SCHEDULERS):
            register_scheduler("dummy", LinearLR)
            self.assertTrue("dummy" in SCHEDULERS)

        self.assertTrue("dummy" not in SCHEDULERS)

    def test_get_scheduler(self):
        with patch.dict(SCHEDULERS):
            s = get_scheduler("LinearLR")
            self.assertEqual(s, LinearLR)

        self.assertTrue("dummy" not in SCHEDULERS)


if __name__ == "__main__":
    unittest.main()
