"""
Predefined constants that will not change and might be used at different places.
"""

import logging
import os

import matplotlib.colors as mcolors
import torch as t

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

OKS_SIGMAS: dict[str, t.Tensor] = {
    # pylint: disable=line-too-long
    # fmt: off
    "halpe-full-body": t.div(
        t.tensor(
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
    "coco-whole-body": t.tensor(
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
    "halpe": t.div(
        t.tensor(
            [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89, 0.8,
             0.8, 0.8, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89]
        ).float(), 10
    ),
    "coco": t.div(
        t.tensor(
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
    "PT21": [
        (16, 14),  #
        (14, 12),  #
        (17, 15),  #
        (15, 13),  #
        (12, 13),  #
        (6, 12),  #
        (7, 13),  #
        (6, 7),  #
        (6, 8),  #
        (7, 9),  #
        (8, 10),  #
        (9, 11),  #
        (2, 3),  #
        (1, 2),  #
        (1, 3),  #
        (2, 4),  #
        (3, 5),  #
        (4, 6),  #
        (5, 7),  #
    ],
}

COLORS: tuple[str, ...] = tuple(mcolors.CSS4_COLORS.keys())

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
    "PT21": [
        "nose",  # 0
        "head_bottom",  # 1
        "head_top",  # 2
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

PRECISION_MAP: [str, t.dtype] = {
    "bfloat16": t.bfloat16,
    "bool": t.bool,
    "cfloat": t.cfloat,
    "complex128": t.complex128,
    "complex64": t.complex64,
    "double": t.double,
    "float": t.float,
    "float16": t.float16,
    "float32": t.float32,
    "float64": t.float64,
    "half": t.half,
    "int": t.int,
    "int16": t.int16,
    "int32": t.int32,
    "int64": t.int64,
    "int8": t.int8,
    "long": t.long,
    "short": t.short,
    "uint8": t.uint8,
}

IMAGE_FORMATS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
"""A list of image formats supported by torchvision."""

# pylint: disable=line-too-long
# fmt: off
VIDEO_FORMATS: tuple[str, ...] = (
    ".264", ".265", ".3g2", ".3gp", ".A64", ".a64", ".adp", ".amr", ".amv", ".asf", ".avc", ".avi", ".avr", ".avs",
    ".avs2", ".avs3", ".bmv", ".cavs", ".cdg", ".cdxl", ".cgi", ".chk", ".cif", ".cpk", ".dat", ".dav", ".dif",
    ".dnxhd", ".dnxhr", ".drc", ".dv", ".dvd", ".f4v", ".flm", ".flv", ".gsm", ".gxf", ".h261", ".h263", ".h264",
    ".h265", ".h26l", ".hevc", ".idf", ".ifv", ".imx", ".ipu", ".ism", ".isma", ".ismv", ".ivf", ".ivr", ".j2k",
    ".kux", ".lvf", ".m1v", ".m2t", ".m2ts", ".m2v", ".m4a", ".m4b", ".m4v", ".mj2", ".mjpeg", ".mjpg", ".mk3d",
    ".mka", ".mks", ".mkv", ".mods", ".moflex", ".mov", ".mp4", ".mpc", ".mpd", ".mpeg", ".mpg", ".mpo", ".mts",
    ".mvi", ".mxf", ".mxg", ".nut", ".obu", ".ogg", ".ogv", ".psp", ".qcif", ".rcv", ".rgb", ".rm", ".roq", ".sdr2",
    ".ser", ".sga", ".svag", ".svs", ".ts", ".ty", ".ty+", ".v", ".v210", ".vb", ".vc1", ".vc2", ".viv", ".vob",
    ".webm", ".wmv", ".wtv", ".xl", ".xmv", ".y4m", ".yop", ".yuv", ".yuv10"
)
"""A list of video formats supported by torchvision. Which uses 'pyav' which is based on 'ffmpeg'."""
# fmt: on

PT21_CATEGORIES: list[dict] = [
    {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": KEY_POINT_NAMES["PT21"],
        "skeleton": [list(pair) for pair in SKELETONS["PT21"]],
    }
]

MODULE_TYPES: tuple[str, ...] = (
    "combine",
    "dataset",
    "dataloader",
    "dgs",
    "embedding_generator",
    "engine",
    "loss",
    "metric",
    "optimizer",
    "similarity",
    "submission",
)
