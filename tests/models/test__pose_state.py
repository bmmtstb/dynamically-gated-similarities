import unittest

import torch
from easydict import EasyDict

from dgs.models.states import PoseState, PoseStates
from dgs.utils.types import Device, PoseStateTuple
from tests.helper import test_multiple_devices

MAX_LENGTH: int = 30
K: int = 15
DIM: int = 2
BBOX_SHAPE: int = 4


def _pose_state_tuple(multiplier: int | float, device: Device, dtype: torch.dtype = None) -> PoseStateTuple:
    """
    Create a tuple of three torch tensors, which will be used as PoseStateTuple.

    Args:
        multiplier: default to tensors with ones but can be multiplied arbitrarily
        device: torch device to send tensor to
        dtype: change dtype of tensor from int to something else

    Returns:
        Tuple of three torch tensors with different shapes but same values.
    """
    return (
        torch.ones((K, DIM), dtype=dtype if dtype else torch.int, device=device) * multiplier,
        torch.ones((K, 1), dtype=dtype if dtype else torch.int, device=device) * multiplier,
        # test with one and two dimensions, even though there should always be two
        torch.ones(BBOX_SHAPE, dtype=dtype if dtype else torch.int, device=device) * multiplier,
    )


def _set_up_default_states(device: Device) -> tuple[PoseStates, PoseStates, PoseStates]:
    """
    Create empty and full pose states object on the given device.

    Args:
        device: The torch device or respective string.

    Returns:
        Three different `PoseStates` objects with zero, one, and maximum number of states.
        zero_states: An empty `PoseStates` object.
        one_states: The tensors have negative float values of -1.
        full_states: The tensors have positive integer values of one times index.
    """
    cfg = EasyDict({"device": device})

    empty_states: PoseStates = PoseStates(config=cfg, max_length=MAX_LENGTH)

    one_states: PoseStates = PoseStates(config=cfg, max_length=MAX_LENGTH)
    one_states.append(_pose_state_tuple(float(-1), device=device, dtype=torch.float))

    # init empty and add values
    full_states: PoseStates = PoseStates(config=cfg, max_length=MAX_LENGTH)
    for i in range(MAX_LENGTH):
        full_states.append(_pose_state_tuple(int(i), device=device))
    return empty_states, one_states, full_states


class TestPoseState(unittest.TestCase):
    @test_multiple_devices
    def test_pose_state_to_device(self, device: Device):
        ps: PoseState = PoseState(
            torch.ones((10, 2)).to("cpu"),
            torch.ones((10, 1)).to("cpu"),
            torch.ones(4).to("cpu"),
        )
        ps_on_device: PoseState = ps.to(device)
        for i in ["pose", "jcs", "bbox"]:
            with self.subTest(msg=f"value: {i}, device: {device}"):
                self.assertEqual(ps_on_device[i].device, torch.empty(1).to(device).device)


class TestPoseStates(unittest.TestCase):
    @test_multiple_devices
    def test_pose_state_len(self, device: Device):
        empty_states, one_states, full_states = _set_up_default_states(device=device)

        for state, length, msg in [
            (empty_states, 0, "empty"),
            (one_states, 1, "one"),
            (full_states, MAX_LENGTH, "full"),
        ]:
            with self.subTest(msg=f"state: {msg}, device: {device}"):
                self.assertEqual(len(state), length)

    @test_multiple_devices
    def test_getitem(self, device: Device):
        _, one_states, full_states = _set_up_default_states(device=device)
        for state, i, expected_state, msg in [
            (one_states, 0, _pose_state_tuple(-1, device, torch.float), "ones first"),
            (one_states, -1, _pose_state_tuple(-1, device, torch.float), "ones negative index"),
            (full_states, 0, _pose_state_tuple(0, device), "full first"),
            (full_states, MAX_LENGTH - 1, _pose_state_tuple(MAX_LENGTH - 1, device), "full last"),
        ]:
            with self.subTest(msg=f"state: {msg}, device: {device}"):
                self.assertEqual(state[i], expected_state)

    @test_multiple_devices
    def test_pose_state_append(self, device: Device):
        empty_states, one_states, full_states = _set_up_default_states(device=device)

        new_state: PoseState = PoseState(
            torch.ones((K, DIM), dtype=torch.float), torch.ones(K), torch.ones((BBOX_SHAPE, 1))
        )
        # Vary dimensions in comparison to _pose_state_tuple()
        # Don't specify the target device, to test whether the append function modifies it correctly.
        # During test time, make sure tensors are on the same device, just don't modify the append method.
        for state, new_length_1, new_length_2, new_first_item, msg in [
            (empty_states, 1, 2, new_state.to(device=device), "empty"),
            (one_states, 2, 3, _pose_state_tuple(-1, device, torch.float), "one"),
            (full_states, MAX_LENGTH, MAX_LENGTH, _pose_state_tuple(2, device), "full"),
        ]:
            with self.subTest(msg=f"state: {msg}, device: {device}"):
                state.append(new_state)
                self.assertEqual(state[-1], new_state.to(device=device), msg="check new item is last")
                self.assertEqual(len(state), new_length_1, msg="check length the first time")

                # append a second time
                state.append(new_state)
                self.assertEqual(
                    state[-2],
                    new_state.to(device=device),
                    msg="doublecheck that previous last item is not second to last",
                )
                self.assertEqual(state[-1], new_state.to(device=device), msg="check last item")
                self.assertEqual(len(state), new_length_2, msg="check length the second time")
                self.assertEqual(state[0], new_first_item, msg="Check the first item")


if __name__ == "__main__":
    unittest.main()
