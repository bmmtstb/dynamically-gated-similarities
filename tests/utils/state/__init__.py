"""
Test values for single and batched State data.
"""

import os

import torch as t
from torchvision import tv_tensors as tvte

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.types import FilePath, FilePaths, Heatmap, Image, Images
from tests.helper import load_test_image

__all__ = [
    "J",
    "J_DIM",
    "B",
    "PID",
    "PIDS",
    "IMG_NAME",
    "DUMMY_IMG",
    "DUMMY_IMG_BATCH",
    "DUMMY_IMGS",
    "DUMMY_KP_TENSOR",
    "DUMMY_KP",
    "DUMMY_KP_BATCH",
    "DUMMY_KP_PATH",
    "DUMMY_KP_PATH_2D",
    "DUMMY_KP_PATH_GLOB",
    "DUMMY_BBOX_TENSOR",
    "DUMMY_BBOX",
    "DUMMY_BBOX_BATCH",
    "DUMMY_FP_STRING",
    "DUMMY_FP",
    "DUMMY_FP_BATCH",
    "DUMMY_HM_TENSOR",
    "DUMMY_HM",
    "DUMMY_HM_BATCH",
    "DUMMY_WEIGHT",
    "DUMMY_WEIGHT_BATCH",
    "DUMMY_DATA",
    "DUMMY_DATA_BATCH",
]

J = 17
J_DIM = 2
B = 2

PID = t.tensor([13], dtype=t.long)
PIDS = t.ones(B, dtype=t.long) * PID

IMG_NAME = "866-200x300.jpg"
DUMMY_IMG: Image = load_test_image(IMG_NAME)
DUMMY_IMG_BATCH: Image = tvte.Image(t.cat([DUMMY_IMG.clone() for _ in range(B)]))
DUMMY_IMGS: Images = [DUMMY_IMG.clone() for _ in range(B)]

# saved pt files include weights (1 x 17 x 3)
DUMMY_KP_PATH: FilePath = os.path.join(PROJECT_ROOT, "./tests/test_data/images/11_1.pt")
DUMMY_KP_PATH_2D: FilePath = os.path.join(PROJECT_ROOT, "./tests/test_data/kp_2d.pt")
DUMMY_KP_PATH_GLOB: FilePath = os.path.join(PROJECT_ROOT, "./tests/test_data/images/11_1_glob.pt")

# plain key points
DUMMY_KP_TENSOR: t.Tensor = t.rand((1, J, J_DIM), dtype=t.float32)
DUMMY_KP = DUMMY_KP_TENSOR.detach().clone()
DUMMY_KP_BATCH: t.Tensor = t.cat([DUMMY_KP_TENSOR.detach().clone() for _ in range(B)])

DUMMY_BBOX_TENSOR: t.Tensor = t.ones((1, 4))
DUMMY_BBOX: tvte.BoundingBoxes = tvte.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tvte.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)
DUMMY_BBOX_BATCH: tvte.BoundingBoxes = tvte.BoundingBoxes(
    t.cat([DUMMY_BBOX_TENSOR.detach().clone() for _ in range(B)]),
    format=tvte.BoundingBoxFormat.XYWH,
    canvas_size=(1000, 1000),
)

DUMMY_FP_STRING: str = os.path.normpath(os.path.join(PROJECT_ROOT, f"./tests/test_data/images/{IMG_NAME}"))
DUMMY_FP: FilePaths = (DUMMY_FP_STRING,)
DUMMY_FP_BATCH: FilePaths = tuple(os.path.normpath(os.path.join(PROJECT_ROOT, DUMMY_FP_STRING)) for _ in range(B))

DUMMY_HM_TENSOR: t.Tensor = t.distributions.uniform.Uniform(0, 1).sample(t.Size((1, J, B, 20))).float()
DUMMY_HM: Heatmap = tvte.Mask(DUMMY_HM_TENSOR.detach().clone(), dtype=t.float32)
DUMMY_HM_BATCH: Heatmap = tvte.Mask(t.cat([DUMMY_HM_TENSOR.detach().clone() for _ in range(B)]), dtype=t.float32)

DUMMY_WEIGHT: t.Tensor = t.tensor([i / J for i in range(J)]).view((1, J, 1)).float()
DUMMY_WEIGHT_BATCH: t.Tensor = t.cat([DUMMY_WEIGHT.detach().clone() for _ in range(B)]).float()

DUMMY_DATA: dict[str, any] = {
    "filepath": DUMMY_FP,
    "crop_path": DUMMY_FP,
    "bbox": DUMMY_BBOX,
    "keypoints": DUMMY_KP,
    "keypoints_local": DUMMY_KP,
    "heatmap": DUMMY_HM,
    "image": [DUMMY_IMG],
    "image_crop": DUMMY_IMG,
    "image_id": PID,
    "frame_id": PID,
    "person_id": PID,
    "class_id": PID,
    "track_id": PID,
    "joint_weight": DUMMY_WEIGHT,
    "pred_tid": PID,
}

DUMMY_DATA_BATCH: dict[str, any] = {
    "filepath": DUMMY_FP_BATCH,
    "crop_path": DUMMY_FP_BATCH,
    "bbox": DUMMY_BBOX_BATCH,
    "keypoints": DUMMY_KP_BATCH,
    "keypoints_local": DUMMY_KP_BATCH,
    "heatmap": DUMMY_HM_BATCH,
    "image": DUMMY_IMGS,
    "image_crop": DUMMY_IMG_BATCH,
    "image_id": PIDS,
    "frame_id": PIDS,
    "person_id": PIDS,
    "class_id": PIDS,
    "track_id": PIDS,
    "joint_weight": DUMMY_WEIGHT_BATCH,
    "pred_tid": PIDS,
}
