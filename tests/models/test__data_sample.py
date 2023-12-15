import os
import unittest

import torch
from torchvision import tv_tensors

from dgs.models.states import DataSample
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.image import load_image

DUMMY_KEY_POINTS_TENSOR: torch.Tensor = torch.rand((1, 20, 2))
DUMMY_KEY_POINTS = DUMMY_KEY_POINTS_TENSOR.detach().clone()

DUMMY_BBOX_TENSOR: torch.Tensor = torch.ones((1, 4)) * 10
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)

DUMMY_FILE_PATH: str = os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"))


class TestDataSample(unittest.TestCase):
    def test_init_regular(self):
        for fp, bbox, kp, out_fp, out_bbox, out_kp in [
            (
                os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"),
                DUMMY_BBOX,
                DUMMY_KEY_POINTS,
                DUMMY_FILE_PATH,
                DUMMY_BBOX,
                DUMMY_KEY_POINTS,
            ),
            (
                "./tests/test_data/866-200x300.jpg",
                DUMMY_BBOX,
                DUMMY_KEY_POINTS_TENSOR,
                DUMMY_FILE_PATH,
                DUMMY_BBOX,
                DUMMY_KEY_POINTS,
            ),
        ]:
            with self.subTest(msg=f"fp: {fp}, bbox: {bbox}, kp: {kp}"):
                ds = DataSample(filepath=fp, bbox=bbox, keypoints=kp)
                self.assertEqual(ds.filepath, tuple([out_fp]))
                self.assertTrue(torch.allclose(ds.bbox, out_bbox))
                self.assertTrue(torch.allclose(ds.keypoints, out_kp))

    def test_get_original_image(self):
        fp = DUMMY_FILE_PATH
        img = load_image(filepath=fp)
        ds1 = DataSample(filepath=fp, bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS)
        ds2 = DataSample(filepath=fp, bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS, original=img)
        self.assertTrue(torch.allclose(ds1.image, ds2.image))

    def test_cast_joint_weight(self):
        pass


if __name__ == "__main__":
    unittest.main()
