"""
Predefined constants that will not change and might be used at different places.
"""
import os

PRINT_PRIORITY: list[str] = ["none", "normal", "debug", "all"]

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

BBOX_FORMATS: list[str] = ["xyxy", "xyah", "xywh", "yolo"]

# connection from -> to with a given color
JOINT_CONNECTIONS_ITOP: list[tuple[int, int, str]] = [
    (0, 1, "b"),  # head neck
    (1, 8, "b"),  # neck torso
    (8, 9, "g"),  # torso rhip
    (9, 11, "g"),  # rhip rknee
    (11, 13, "g"),  # rknee rfoot
    (8, 10, "m"),  # torso lhip
    (10, 12, "m"),  # lhip lknee
    (12, 14, "m"),  # lknee lfoot
    (1, 2, "y"),  # neck rshoulder
    (2, 4, "y"),  # rshoulder rellbow
    (4, 6, "y"),  # rellbow rhand
    (1, 3, "c"),  # neck lshoulder
    (3, 5, "c"),  # lshoulder lellbow
    (5, 7, "c"),  # lellbow lhand
]
