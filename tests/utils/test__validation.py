import os
import unittest

import numpy as np
import torch
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.validation import (
    validate_bboxes,
    validate_dimensions,
    validate_filepath,
    validate_images,
    validate_key_points,
)

DUMMY_IMAGE_TENSOR: torch.ByteTensor = torch.ByteTensor(torch.ones((1, 3, 10, 20), dtype=torch.uint8))
DUMMY_IMAGE: tv_tensors.Image = tv_tensors.Image(DUMMY_IMAGE_TENSOR)

DUMMY_KEY_POINTS_TENSOR: torch.Tensor = torch.rand((1, 20, 2))
DUMMY_KEY_POINTS: tv_tensors.Mask = tv_tensors.Mask(DUMMY_KEY_POINTS_TENSOR)

DUMMY_BBOX_TENSOR: torch.Tensor = torch.ones((1, 4)) * 10
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)


class TestDataSampleValidation(unittest.TestCase):
    def test_validate_image(self):
        for image, dims, output in [
            (DUMMY_IMAGE_TENSOR, 4, DUMMY_IMAGE),
            (DUMMY_IMAGE_TENSOR, None, DUMMY_IMAGE),
            (DUMMY_IMAGE_TENSOR.squeeze_(), 4, DUMMY_IMAGE),
            (
                DUMMY_IMAGE_TENSOR.squeeze_(),
                None,
                tv_tensors.Image(torch.ByteTensor(torch.ones((3, 10, 20), dtype=torch.uint8))),
            ),
            (torch.ByteTensor(torch.ones((1, 1, 1, 3, 10, 20), dtype=torch.uint8)), 4, DUMMY_IMAGE),
            (
                torch.ByteTensor(torch.ones((1, 1, 1, 3, 10, 20), dtype=torch.uint8)),
                None,
                tv_tensors.Image(torch.ByteTensor(torch.ones((1, 1, 1, 3, 10, 20), dtype=torch.uint8))),
            ),
        ]:
            with self.subTest(msg=f"image: {image}, dims: {dims}"):
                self.assertTrue(
                    torch.allclose(
                        validate_images(images=image, dims=dims),
                        output,
                    )
                )

    def test_validate_images_exceptions(self):
        for image, dims, exception_type in [
            (np.ones((1, 3, 10, 20), dtype=int), None, TypeError),
            (torch.BoolTensor(torch.ones((1, 3, 10, 20), dtype=torch.bool)), None, TypeError),
            (torch.ByteTensor(torch.ones((1, 2, 10, 20), dtype=torch.uint8)), None, ValueError),
            (torch.ByteTensor(torch.ones((1, 10, 10, 20), dtype=torch.uint8)), None, ValueError),
            (torch.ByteTensor(torch.ones((1, 0, 10, 20), dtype=torch.uint8)), None, ValueError),
            (torch.ByteTensor(torch.ones((10, 20), dtype=torch.uint8)), None, ValueError),
            (torch.ByteTensor(torch.ones(1000, dtype=torch.uint8)), None, ValueError),
        ]:
            with self.subTest(msg=f"image: {image}, dims: {dims}"):
                with self.assertRaises(exception_type):
                    validate_images(images=image, dims=dims)

    def test_validate_key_points(self):
        for key_points, dims, nof_joints, joint_dim, output in [
            (DUMMY_KEY_POINTS_TENSOR, 3, 20, 2, DUMMY_KEY_POINTS),
            (DUMMY_KEY_POINTS, 3, 20, 2, DUMMY_KEY_POINTS),
            (DUMMY_KEY_POINTS_TENSOR, None, None, None, DUMMY_KEY_POINTS),
            (DUMMY_KEY_POINTS, None, None, None, DUMMY_KEY_POINTS),
            (torch.ones((1, 10, 3)), 3, None, None, tv_tensors.Mask(torch.ones((1, 10, 3)))),  # test 3d
            (torch.ones((1, 10, 3)), None, None, None, tv_tensors.Mask(torch.ones((1, 10, 3)))),
            (torch.ones((10, 3)), 3, None, None, tv_tensors.Mask(torch.ones((1, 10, 3)))),  # make more dims
            (
                torch.ones((1, 1, 1, 10, 3)),
                3,
                None,
                None,
                tv_tensors.Mask(torch.ones((1, 10, 3))),
            ),  # reduce the amount of dims
        ]:
            with self.subTest(msg=f"key points: {key_points}, dims: {dims}"):
                self.assertTrue(
                    torch.allclose(
                        validate_key_points(
                            key_points=key_points, dims=dims, joint_dim=joint_dim, nof_joints=nof_joints
                        ),
                        output,
                    )
                )

    def test_validate_key_points_exception(self):
        for key_points, dims, nof_joints, joint_dim, exception_type in [
            (np.ones((1, 10, 2), dtype=int), None, None, None, TypeError),
            (torch.ones(1, 100, 1), None, None, None, ValueError),
            (torch.ones(10, 100, 4), None, None, None, ValueError),
            (torch.ones(10, 100, 4), None, None, None, ValueError),
            (torch.ones(1, 100, 2), None, 50, None, ValueError),
            (torch.ones(1, 100, 2), None, None, 3, ValueError),
        ]:
            with self.subTest(f"dims: {dims}, nof_joints: {nof_joints}, joint_dim: {joint_dim}"):
                with self.assertRaises(exception_type):
                    validate_key_points(key_points=key_points, dims=dims, joint_dim=joint_dim, nof_joints=nof_joints)

    def test_validate_bboxes(self):
        for bboxes, dims, box_format, output in [
            (DUMMY_BBOX, 2, None, DUMMY_BBOX),
            (DUMMY_BBOX, None, None, DUMMY_BBOX),
            (DUMMY_BBOX, None, tv_tensors.BoundingBoxFormat.XYWH, DUMMY_BBOX),
            (
                tv_tensors.BoundingBoxes(torch.ones(4), format="XYWH", canvas_size=(10, 10)),
                2,
                None,
                tv_tensors.BoundingBoxes(torch.ones((1, 4)), format="XYWH", canvas_size=(10, 10)),
            ),
        ]:
            with self.subTest(msg=f"bboxes: {bboxes}, dims: {dims}"):
                self.assertTrue(
                    torch.allclose(
                        validate_bboxes(bboxes=bboxes, dims=dims, box_format=box_format),
                        output,
                    )
                )

    def test_validate_bboxes_exceptions(self):
        for bboxes, dims, box_format, exception_type in [
            (torch.ones((1, 4)), 2, None, TypeError),
            (np.ones((1, 4)), 2, None, TypeError),
            (DUMMY_BBOX, None, tv_tensors.BoundingBoxFormat.XYXY, ValueError),
            (DUMMY_BBOX, None, tv_tensors.BoundingBoxFormat.CXCYWH, ValueError),
        ]:
            with self.subTest():
                with self.assertRaises(exception_type):
                    validate_bboxes(bboxes=bboxes, dims=dims, box_format=box_format),

    def test_validate_dimensions(self):
        for tensor, dims, output in [
            (torch.ones((1, 1)), 2, torch.ones((1, 1))),
            (torch.ones((2, 2, 2)), 3, torch.ones((2, 2, 2))),
            (torch.ones((1, 1, 1, 1, 5)), 1, torch.ones(5)),
            (torch.ones(5), 5, torch.ones((1, 1, 1, 1, 5))),
            (torch.ones((2, 1, 5)), 2, torch.ones((2, 5))),
        ]:
            with self.subTest(msg=f"tensor: {tensor}, dims: {dims}"):
                self.assertTrue(
                    torch.allclose(
                        validate_dimensions(tensor=tensor, dims=dims),
                        output,
                    )
                )

    def test_validate_dimensions_exceptions(self):
        for tensor, dims, exception_type in [
            (np.ones((1, 1)), 1, TypeError),
            (torch.ones((2, 2, 5)), 2, ValueError),
            (torch.ones((2, 1, 5)), 1, ValueError),
        ]:
            with self.subTest():
                with self.assertRaises(exception_type):
                    validate_dimensions(tensor=tensor, dims=dims),

    def test_validate_file_path(self):
        full_path = os.path.normpath(os.path.join(PROJECT_ROOT, "tests/test_data/283-200x300.jpg"))
        for file_path in [
            "./tests/test_data/283-200x300.jpg",
            os.path.join(PROJECT_ROOT, "tests/test_data/283-200x300.jpg"),
        ]:
            with self.subTest(msg=f"file_path: {file_path}"):
                self.assertEqual(
                    validate_filepath(file_path=file_path),
                    full_path,
                )


if __name__ == "__main__":
    unittest.main()
