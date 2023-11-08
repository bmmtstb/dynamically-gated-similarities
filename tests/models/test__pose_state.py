import unittest

import torch

from dgs.models.pose_state import PoseStates
from tests.test__parameterized import DeviceParameterizedTestCase

MAX_LENGTH: int = 30
K: int = 15
DIM: int = 2
BBOX_SHAPE: int = 4


class TestPoseStatesDevice(DeviceParameterizedTestCase):

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)  # make sure to pass down args and kwargs

        self.empty_states: PoseStates = PoseStates(config=self.config, max_length=MAX_LENGTH)

        # init empty and add values
        self.full_states: PoseStates = PoseStates(config=self.config, max_length=MAX_LENGTH)
        for i in range(MAX_LENGTH):
            self.full_states.append((
                torch.ones((K, DIM)) * i,
                torch.ones((K, 1)) * i,
                torch.ones(4) * i,  # test with one and two dimensions
            ))

    def test_pose_state_len(self):
        for state, length, msg in [
            (self.empty_states, 0, "empty"),
            (self.full_states, MAX_LENGTH, "full"),
        ]:
            with self.subTest(msg=f'state: {msg}'):
                self.assertEqual(len(state), length)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(DeviceParameterizedTestCase.parametrize(TestPoseStatesDevice, device="cpu"))
    suite.addTest(DeviceParameterizedTestCase.parametrize(TestPoseStatesDevice, device="cuda"))
    unittest.TextTestRunner(verbosity=2).run(suite)
