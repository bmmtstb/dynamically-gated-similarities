import os
import unittest

import numpy as np
import torch
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException, ValidationException
from dgs.utils.validation import (
    validate_bboxes,
    validate_dimensions,
    validate_filepath,
    validate_heatmaps,
    validate_ids,
    validate_images,
    validate_key_points,
    validate_value,
)

J = 20

DUMMY_IMAGE_TENSOR: torch.ByteTensor = torch.ones((1, 3, 10, 20)).byte()
DUMMY_IMAGE: tv_tensors.Image = tv_tensors.Image(DUMMY_IMAGE_TENSOR)

DUMMY_KEY_POINTS_TENSOR: torch.Tensor = torch.rand((1, J, 2))
DUMMY_KEY_POINTS: torch.Tensor = DUMMY_KEY_POINTS_TENSOR.detach().clone()

DUMMY_BBOX_TENSOR: torch.Tensor = torch.ones((1, 4)) * 10
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)

DUMMY_HM_TENSOR: torch.FloatTensor = (
    torch.distributions.uniform.Uniform(0, 1).sample(torch.Size((1, J, 10, 20))).float()
)
DUMMY_HM: tv_tensors.Mask = tv_tensors.Mask(DUMMY_HM_TENSOR, dtype=torch.float32)


class TestValidation(unittest.TestCase):
    def test_validate_image(self):
        for image, dims, output in [
            (DUMMY_IMAGE_TENSOR, 4, DUMMY_IMAGE),
            (DUMMY_IMAGE_TENSOR, None, DUMMY_IMAGE),
            (DUMMY_IMAGE_TENSOR.squeeze_(), 4, DUMMY_IMAGE),
            (
                DUMMY_IMAGE_TENSOR.squeeze_(),
                None,
                tv_tensors.Image(torch.ones((3, 10, 20)).byte()),
            ),
            (torch.ones((1, 1, 1, 3, 10, 20)).byte(), 4, DUMMY_IMAGE),
            (
                torch.ones((1, 1, 1, 3, 10, 20)).byte(),
                None,
                tv_tensors.Image(torch.ones((1, 1, 1, 3, 10, 20)).byte()),
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
            (torch.ones((1, 3, 10, 20)).bool(), None, TypeError),  # bool tensor
            (torch.ones((1, 2, 10, 20)).byte(), None, ValueError),
            (torch.ones((1, 10, 10, 20)).byte(), None, ValueError),
            (torch.ones((1, 0, 10, 20)).byte(), None, ValueError),
            (torch.ones((10, 20)).byte(), None, ValueError),
            (torch.ones(1000).byte(), None, ValueError),
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
            (torch.ones((1, 10, 3)), 3, None, None, torch.ones((1, 10, 3))),  # test 3d
            (torch.ones((1, 10, 3)), None, None, None, torch.ones((1, 10, 3))),
            (torch.ones((10, 3)), 3, None, None, torch.ones((1, 10, 3))),  # make more dims
            (
                torch.ones((1, 1, 1, 10, 3)),
                3,
                None,
                None,
                torch.ones((1, 10, 3)),
            ),  # reduce the number of dimensions
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
            (np.ones((2, 5), dtype=np.int32), 2, torch.ones(size=(2, 5), dtype=torch.int32)),
            ([[1, 1]], 2, torch.tensor([[1, 1]])),
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
            ([[1, 1], [1]], 1, TypeError),
            ("dummy", 1, TypeError),
            (torch.ones((2, 2, 5)), 2, ValueError),
            (torch.ones((2, 1, 5)), 1, ValueError),
        ]:
            with self.subTest(msg=f"tensor: {tensor}, dims: {dims}, exception_type={exception_type}"):
                with self.assertRaises(exception_type):
                    validate_dimensions(tensor=tensor, dims=dims),

    def test_validate_file_path(self):
        full_path = tuple([os.path.normpath(os.path.join(PROJECT_ROOT, "tests/test_data/866-200x300.jpg"))])
        for file_path in [
            "./tests/test_data/866-200x300.jpg",
            os.path.join(PROJECT_ROOT, "tests/test_data/866-200x300.jpg"),
            ["./tests/test_data/866-200x300.jpg"],
        ]:
            with self.subTest(msg=f"file_path: {file_path}"):
                self.assertEqual(
                    validate_filepath(file_paths=file_path),
                    full_path,
                )

    def test_validate_filepath_exceptions(self):
        for fps, exception_type in [
            ("", InvalidPathException),
            (os.path.join(PROJECT_ROOT, "dummy"), InvalidPathException),
        ]:
            with self.subTest(msg=f"filepath: {fps}, exception_type={exception_type}"):
                with self.assertRaises(exception_type):
                    validate_filepath(file_paths=fps),

    def test_validate_ids(self):
        for tensor, output in [
            (1, torch.ones(1).to(dtype=torch.int32)),
            (123456, torch.tensor([123456]).int()),
            (torch.ones((1, 1, 100, 1)).int(), torch.ones(1).to(dtype=torch.int32)),
            (torch.ones((2, 1)).int(), torch.ones(2).to(dtype=torch.int32)),
            (torch.ones(20).int(), torch.ones(20).to(dtype=torch.int32)),
            (torch.tensor([[1, 2, 3, 4]]).int(), torch.tensor([1, 2, 3, 4]).int()),
        ]:
            with self.subTest(msg=f"ids: {tensor}"):
                self.assertTrue(torch.allclose(validate_ids(ids=tensor), output))

    def test_validate_ids_exceptions(self):
        for tensor, exception_type in [
            (np.ones((1, 10)), TypeError),
            (torch.ones((2, 5)).float(), TypeError),
            (torch.ones((2, 5)).int(), ValueError),
            (torch.ones((1, 2, 1, 5)).int(), ValueError),
        ]:
            with self.subTest():
                with self.assertRaises(exception_type):
                    validate_ids(ids=tensor),

    def test_validate_heatmaps(self):
        for tensor, dims, n_j, output in [
            (DUMMY_HM_TENSOR, None, None, DUMMY_HM),
            (torch.zeros((J, 10, 20)), 4, J, tv_tensors.Mask(torch.zeros((1, J, 10, 20)))),
            (torch.zeros((1, 1, J, 10, 20)), 3, J, tv_tensors.Mask(torch.zeros((J, 10, 20)))),
        ]:
            with self.subTest(msg=f"heatmap: {tensor}"):
                self.assertTrue(torch.allclose(validate_heatmaps(tensor, dims=dims, nof_joints=n_j), output))

    def test_validate_heatmaps_exceptions(self):
        for tensor, n_j, exception_type in [
            (np.ones((1, 10)), None, TypeError),
            (torch.ones((2, 5)), J, ValueError),
            (torch.ones((J + 1, 2, 5)), J, ValueError),
        ]:
            with self.subTest(f"shape: {tensor.shape}, n_j: {n_j}, excp: {exception_type}"):
                with self.assertRaises(exception_type):
                    validate_heatmaps(tensor, nof_joints=n_j),


class TestValidateValue(unittest.TestCase):
    def test_validate_value(self):
        for value, data, validation, result in [
            (None, ..., "None", True),
            (None, ..., "not None", False),
            (1, ..., "None", False),
            (1, ..., "not None", True),
            (1, (1, 2, 3), "in", True),
            (1.5, (1, 2, 3), "in", False),
            (1, [1, 2, 3], "in", True),
            ("1", [1, 2, 3], "in", False),
            (1, ["1", "2", "3"], "in", False),
            ("1", ["1", "2", "3"], "in", True),
            (1, ..., "float", False),
            (1.0, ..., "float", True),
            (1, float, "instance", False),
            (1.0, float, "instance", True),
            (["1", "2", "3"], 1, "contains", False),
            (["1", "2", "3"], "1", "contains", True),
            (..., ..., "optional", True),
        ]:
            with self.subTest(msg=f"value: {value}, data: {data}, validation: {validation}"):
                self.assertEqual(validate_value(value, data, validation), result)

    def test_validate_value_raises(self):
        for value, data, validation, exception_type in [
            (None, None, "dummy", KeyError),
            ([1, 2, 3], "a", "shorter", ValidationException),
        ]:
            with self.subTest(msg=f"value: {value}, data: {data}, validation: {validation}"):
                with self.assertRaises(exception_type):
                    validate_value(value, data, validation),

    def test_nested_validations(self):
        for value, data, validation, valid in [
            ("cuda", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", True),
            ("cpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", True),
            ("gpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", False),
            (torch.device("cuda"), (("in", ["cuda", "cpu"]), ("instance", torch.device)), "or", True),
            ("cuda", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", True),
            ("cpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", True),
            ("gpu", (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", False),
            (torch.device("cuda"), (("in", ["cuda", "cpu"]), ("instance", torch.device)), "xor", True),
            (None, (("None", ...), ("not", ("not None", ...))), "and", True),
            (1, (("gte", 1), ("not None", ...), ("lte", 1.1), ("eq", 1), ("int", ...)), "and", True),
        ]:
            with self.subTest(msg=f"value {value}, validation {validation}"):
                self.assertEqual(validate_value(value, data, validation), valid)


if __name__ == "__main__":
    unittest.main()
