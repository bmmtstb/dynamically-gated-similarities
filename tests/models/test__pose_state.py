import unittest

import torch


class TestPoseStates(unittest.TestCase):

    def __init__(self, device: torch.device | str) -> None:
        """
        test pose state on different devices

        Args:
            device: device to test on
        """
        super().__init__()
        self.device = device

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestPoseStates('cuda'))
    suite.addTest(TestPoseStates('cpu'))
    unittest.TextTestRunner().run(suite)
