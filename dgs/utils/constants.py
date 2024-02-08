"""
Predefined constants that will not change and might be used at different places.
"""

import logging
import os

import torch

PRINT_PRIORITY: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
"""Stringified names of all the available print / logging priorities."""


PROJECT_ROOT: str = os.path.normpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
"""Path to this projects' root directory."""

OKS_SIGMAS: dict[str, torch.FloatTensor] = {
    # pylint: disable=line-too-long
    # fmt: off
    "halpe-full-body": torch.div(
        torch.tensor(
            [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.8,
             0.8, 0.8, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
             0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
             0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
             0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
             0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
             0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
             0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        ).float(), 10
    ),
    "coco-whole-body": torch.tensor(
        [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089,
         0.089, 0.068, 0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025, 0.020,
         0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011,
         0.013, 0.015, 0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011, 0.009,
         0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009,
         0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01, 0.008, 0.029, 0.022, 0.035, 0.037, 0.047,
         0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031,
         0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021,
         0.032, 0.02, 0.019, 0.022, 0.031]
    ).float(),
    "halpe": torch.div(
        torch.tensor(
            [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.8,
             0.8, 0.8, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89]
        ).float(), 10
    ),
    "coco": torch.div(
        torch.tensor(
            [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]
        ).float(), 10
    ),
    # fmt: on
}

SKELETONS: dict[str, list[tuple[int, int]]] = {
    "coco": [
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
    ],
    "coco_no_ears": [
        (0, 1),  # nose l_eye
        (0, 2),  # nose r_eye
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
    ],
}

KEY_POINT_NAMES: dict[str, list[str]] = {
    "coco": [
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
    ],
}
