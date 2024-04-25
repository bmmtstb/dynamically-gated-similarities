"""
Test values for single and batched State data.
"""

import os

import torch
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.types import FilePaths, Heatmap, Image, Images
from tests.helper import load_test_image

J = 17
J_DIM = 2
B = 2

PID = torch.tensor([13], dtype=torch.long)
PIDS = torch.ones(B, dtype=torch.long) * PID

IMG_NAME = "866-200x300.jpg"
DUMMY_IMG: Image = load_test_image(IMG_NAME)
DUMMY_IMG_BATCH: Image = tv_tensors.Image(torch.cat([DUMMY_IMG.clone() for _ in range(B)]))
DUMMY_IMGS: Images = [DUMMY_IMG.clone() for _ in range(B)]

DUMMY_KP_TENSOR: torch.Tensor = torch.rand((1, J, J_DIM), dtype=torch.float32)
DUMMY_KP = DUMMY_KP_TENSOR.detach().clone()
DUMMY_KP_BATCH: torch.Tensor = torch.cat([DUMMY_KP_TENSOR.detach().clone() for _ in range(B)])

DUMMY_BBOX_TENSOR: torch.Tensor = torch.ones((1, 4))
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)
DUMMY_BBOX_BATCH: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    torch.cat([DUMMY_BBOX_TENSOR.detach().clone() for _ in range(B)]),
    format=tv_tensors.BoundingBoxFormat.XYWH,
    canvas_size=(1000, 1000),
)

DUMMY_FP_STRING: str = os.path.normpath(os.path.join(PROJECT_ROOT, f"./tests/test_data/images/{IMG_NAME}"))
DUMMY_FP: FilePaths = (DUMMY_FP_STRING,)
DUMMY_FP_BATCH: FilePaths = tuple(os.path.normpath(os.path.join(PROJECT_ROOT, DUMMY_FP_STRING)) for _ in range(B))

DUMMY_HM_TENSOR: torch.Tensor = torch.distributions.uniform.Uniform(0, 1).sample(torch.Size((1, J, B, 20))).float()
DUMMY_HM: Heatmap = tv_tensors.Mask(DUMMY_HM_TENSOR.detach().clone(), dtype=torch.float32)
DUMMY_HM_BATCH: Heatmap = tv_tensors.Mask(
    torch.cat([DUMMY_HM_TENSOR.detach().clone() for _ in range(B)]), dtype=torch.float32
)

DUMMY_WEIGHT: torch.Tensor = torch.tensor([i / J for i in range(J)]).view((1, J, 1)).float()
DUMMY_WEIGHT_BATCH: torch.Tensor = torch.cat([DUMMY_WEIGHT.detach().clone() for _ in range(B)]).float()

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
}
