import os
import shutil
import unittest

import numpy as np
import torch
from torchvision.tv_tensors import BoundingBoxes

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import is_project_file
from dgs.utils.types import Device
from dgs.utils.utils import extract_crops_from_images, torch_to_numpy
from helper import test_multiple_devices


class TestUtils(unittest.TestCase):
    @test_multiple_devices
    def test_torch_to_np(self, device: Device):
        for torch_tensor, numpy_array in [
            (torch.ones((5, 2), device=device, dtype=torch.int32), np.ones((5, 2), dtype=np.int32)),
        ]:
            with self.subTest(msg=f"torch_tensor: {torch_tensor}, numpy_array: {numpy_array}"):
                self.assertTrue(np.array_equal(torch_to_numpy(torch_tensor), numpy_array))

    @test_multiple_devices
    def test_extract_crops_from_images(self, device: Device):
        base_path = os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/"))
        img_src = os.path.normpath(os.path.join(base_path, "./test_data/"))
        crop_target = os.path.normpath(os.path.join(base_path, "./test_crops"))
        for img_shapes, crop_fps, bbox_format, kwargs in [
            ([(300, 200)], ["1.jpg"], "xywh", {"device": device}),
            ([(300, 200), (300, 200)], ["2_1.jpg", "2_2.jpg"], "xyxy", {"device": device}),
            ([(300, 200) for _ in range(10)], [f"3_{i}.jpg" for i in range(10)], "xyxy", {"device": device}),
            ([(500, 1000)], ["4.jpg"], "xywh", {"transform_mode": "mean-pad", "crop_size": (128, 128), "quality": 50}),
        ]:
            with self.subTest(
                msg=f"shapes: {img_shapes}, crop_fps: {crop_fps}, format: {bbox_format}, kwargs: {kwargs}"
            ):
                box_coords = torch.stack([torch.tensor([0, 1, 10, 21]) for _ in range(len(img_shapes))])
                bboxes = BoundingBoxes(box_coords, canvas_size=max(img_shapes), format=bbox_format)
                img_fps = [os.path.join(img_src, f"866-{shape[1]}x{shape[0]}.jpg") for shape in img_shapes]
                crop_fps = [os.path.join(crop_target, c) for c in crop_fps]
                crops, kp = extract_crops_from_images(img_fps=img_fps, new_fps=crop_fps, boxes=bboxes, **kwargs)

                self.assertEqual(len(crops), len(img_shapes))
                self.assertEqual(len(kp), len(img_shapes))
                for fp in crop_fps:
                    self.assertTrue(is_project_file(fp))
        # delete crops folder in the end
        shutil.rmtree(crop_target)

    def test_extract_crops_from_images_exceptions(self):
        bbox = BoundingBoxes(torch.tensor([1, 2, 3, 4]), canvas_size=(100, 100), format="xywh")
        for img_fps, crop_fps, bboxes in [
            (["dummy"], [], bbox),
            ([], ["dummy"], bbox),
            (["dummy"], ["dummy"], BoundingBoxes(torch.ones((2, 4)), canvas_size=(100, 100), format="xywh")),
        ]:
            with self.subTest(msg=f"img_fps, crop_fps, bbox"):
                with self.assertRaises(ValueError):
                    extract_crops_from_images(img_fps=img_fps, new_fps=crop_fps, boxes=bboxes)


if __name__ == "__main__":
    unittest.main()
