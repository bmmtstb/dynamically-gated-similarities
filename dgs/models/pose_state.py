"""
definitions and helpers for pose-state(s)
"""
import torch

from dgs.utils.types import Config, PoseState


class PoseStates:
    """
    A double-ended-queue
    """

    def __init__(self, config: Config, max_length: int = 30):
        self.config: Config = config

        self.max_length: int = max_length  # FIXME: get from params

        self.jcss: list[torch.Tensor] = []
        self.poses: list[torch.Tensor] = []
        self.bboxes: list[torch.Tensor] = []

    def get_states(self) -> PoseState:
        """Obtain a copy of the current states of the deque to ensure equal length and times at all times"""

        def stack_tensor_copy(lot: list[torch.Tensor]) -> torch.Tensor:
            """Stack state and create a copy of the tensor"""
            return torch.stack(lot).detach().clone().to(self.config.device)

        return (
            stack_tensor_copy(self.poses),
            stack_tensor_copy(self.jcss),
            stack_tensor_copy(self.bboxes)
        )

    def __len__(self) -> int:
        """get length of this state"""
        if len(self.jcss) == len(self.poses) == len(self.bboxes):
            return len(self.jcss)
        raise IndexError("Lists in pose state have different shape.")

    def append(self, new_state: PoseState) -> None:
        """
        Right-Append new_state to the current states, but make sure that max length

        Args:
            new_state: pose, jcs and bbox to append to current state
        """
        # pop old state if too long
        if len(self) >= self.max_length:
            self.jcss.pop(0)
            self.poses.pop(0)
            self.bboxes.pop(0)
        # append new state
        jcs, pose, bbox = new_state
        self.jcss.append(jcs)
        self.poses.append(pose)
        self.bboxes.append(bbox)
