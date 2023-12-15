"""
Predefined constants that will not change and might be used at different places.
"""
import os

PRINT_PRIORITY: list[str] = ["none", "normal", "debug", "all"]
"""List of all the available print priorities."""

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
"""Path to project root directory."""

COCO_SKELETON = [
    (0, 1),  # nose l_eye
    (0, 2),  # nose r_eye
    (1, 3),  # l_eye l_ear
    (2, 4),  # r_eye r_ear
    (0, 5),  # nose l_shoulder
    (0, 6),  # nose r_shoulder
    (5, 7),  # l_shoulder l_elbow
    (6, 8),  # r_shoulder r_elbow
    (7, 9),  # l_elbow l_wrist
    (8, 10),  # r_elbow r_wrist
    (5, 11),  # l_shoulder l_hip
    (6, 12),  # r_shoulder r_hip
    (11, 13),  # l_hip l_knee
    (12, 14),  # r_hip r_knee
    (13, 15),  # l_knee l_ankle
    (14, 16),  # r_knee r_ankle
]

COCO_KEY_POINTS = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
]
