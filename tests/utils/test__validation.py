import os
import unittest

import numpy as np
import torch as t
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException, ValidationException
from dgs.utils.types import FilePaths
from dgs.utils.validation import (
    validate_bboxes,
    validate_dimensions,
    validate_filepath,
    validate_heatmaps,
    validate_ids,
    validate_image,
    validate_images,
    validate_key_points,
    validate_value,
)

J = 20
B = 10
IMG_NAME = "866-200x300.jpg"

DUMMY_IMAGE_TENSOR: t.Tensor = t.ones((1, 3, 10, 20)).byte()
DUMMY_IMAGE: tv_tensors.Image = tv_tensors.Image(DUMMY_IMAGE_TENSOR)

DUMMY_KEY_POINTS_TENSOR: t.Tensor = t.rand((1, J, 2))
DUMMY_KEY_POINTS: t.Tensor = DUMMY_KEY_POINTS_TENSOR.detach().clone()

DUMMY_BBOX_TENSOR: t.Tensor = t.ones((1, 4)) * 10
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)

DUMMY_ID: t.Tensor = t.ones(1).long()

DUMMY_HM_TENSOR: t.Tensor = t.distributions.uniform.Uniform(0, 1).sample(t.Size((1, J, 10, 20))).float()
DUMMY_HM: tv_tensors.Mask = tv_tensors.Mask(DUMMY_HM_TENSOR, dtype=t.float32)

DUMMY_FP_STRING: str = f"./tests/test_data/images/{IMG_NAME}"
DUMMY_FP: FilePaths = (os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/test_data/images/" + IMG_NAME)),)
DUMMY_FP_BATCH: FilePaths = tuple(DUMMY_FP_STRING for _ in range(B))


class TestValidateImage(unittest.TestCase):

    def test_validate_image(self):
        for image, dims, output in [
            (DUMMY_IMAGE_TENSOR, 4, DUMMY_IMAGE),
            (DUMMY_IMAGE_TENSOR, None, DUMMY_IMAGE),
            (DUMMY_IMAGE_TENSOR.squeeze_(), 4, DUMMY_IMAGE),
            (
                DUMMY_IMAGE_TENSOR.squeeze_(),
                None,
                tv_tensors.Image(t.ones((3, 10, 20)).byte()),
            ),
            (t.ones((1, 1, 1, 3, 10, 20)).byte(), 4, DUMMY_IMAGE),
            (
                t.ones((1, 1, 1, 3, 10, 20)).byte(),
                None,
                tv_tensors.Image(t.ones((1, 1, 1, 3, 10, 20)).byte()),
            ),
        ]:
            with self.subTest(msg=f"image: {image}, dims: {dims}"):
                self.assertTrue(t.allclose(validate_image(images=image, dims=dims), output))

    def test_validate_image_exceptions(self):
        for image, dims, exception_type in [
            (np.ones((1, 3, 10, 20), dtype=int), None, TypeError),
            (t.ones((1, 3, 10, 20)).bool(), None, TypeError),  # bool tensor
            (t.ones((1, 2, 10, 20)).byte(), None, ValueError),
            (t.ones((1, 10, 10, 20)).byte(), None, ValueError),
            (t.ones((1, 0, 10, 20)).byte(), None, ValueError),
            (t.ones((10, 20)).byte(), None, ValueError),
            (t.ones(1000).byte(), None, ValueError),
        ]:
            with self.subTest(msg=f"image: {image}, dims: {dims}"):
                with self.assertRaises(exception_type):
                    validate_image(images=image, dims=dims)

    def test_img_length(self):
        self.assertTrue(t.allclose(validate_image(DUMMY_IMAGE, length=1, dims=None), DUMMY_IMAGE))

    def test_img_length_exception(self):
        with self.assertRaises(ValidationException) as e:
            _ = validate_image(DUMMY_IMAGE, length=4, dims=None)
        self.assertTrue("Image length is expected to be 4 but got 1" in str(e.exception), msg=e.exception)


class TestValidateImages(unittest.TestCase):

    def test_validate_images(self):
        for images, output in [
            ([], []),
            ([DUMMY_IMAGE], [DUMMY_IMAGE]),
            ([DUMMY_IMAGE_TENSOR], [DUMMY_IMAGE]),
            ([DUMMY_IMAGE, DUMMY_IMAGE_TENSOR], [DUMMY_IMAGE, DUMMY_IMAGE]),
        ]:
            with self.subTest(msg=f"images: {images}"):
                res_img = validate_images(images=images)
                self.assertEqual(len(images), len(output))
                self.assertEqual(len(images), len(res_img))
                self.assertTrue(all(t.allclose(res, out) for res, out in zip(res_img, output)))

    def test_validate_images_exceptions(self):
        for images, exception_type, err_msg in [
            ({DUMMY_IMAGE}, TypeError, "Expected images to be a list, got"),
            (DUMMY_IMAGE, TypeError, "Expected images to be a list, got"),
        ]:
            with self.subTest(msg=f"images: {images}"):
                with self.assertRaises(exception_type):
                    # noinspection PyTypeChecker
                    validate_images(images=images)


class TestValidateKeyPoints(unittest.TestCase):

    def test_validate_key_points(self):
        for key_points, dims, nof_joints, joint_dim, output in [
            (DUMMY_KEY_POINTS_TENSOR, 3, 20, 2, DUMMY_KEY_POINTS),
            (DUMMY_KEY_POINTS, 3, 20, 2, DUMMY_KEY_POINTS),
            (DUMMY_KEY_POINTS_TENSOR, None, None, None, DUMMY_KEY_POINTS),
            (DUMMY_KEY_POINTS, None, None, None, DUMMY_KEY_POINTS),
            (t.ones((1, 10, 3)), 3, None, None, t.ones((1, 10, 3))),  # test 3d
            (t.ones((1, 10, 3)), None, None, None, t.ones((1, 10, 3))),
            (t.ones((10, 3)), 3, None, None, t.ones((1, 10, 3))),  # make more dims
            (
                t.ones((1, 1, 1, 10, 3)),
                3,
                None,
                None,
                t.ones((1, 10, 3)),
            ),  # reduce the number of dimensions
        ]:
            with self.subTest(msg=f"key points: {key_points}, dims: {dims}"):
                self.assertTrue(
                    t.allclose(
                        validate_key_points(
                            key_points=key_points, dims=dims, joint_dim=joint_dim, nof_joints=nof_joints
                        ),
                        output,
                    )
                )

    def test_validate_key_points_exception(self):
        for key_points, dims, nof_joints, joint_dim, exception_type in [
            (np.ones((1, 10, 2), dtype=int), None, None, None, TypeError),
            (t.ones(1, 100, 1), None, None, None, ValueError),
            (t.ones(10, 100, 4), None, None, None, ValueError),
            (t.ones(10, 100, 4), None, None, None, ValueError),
            (t.ones(1, 100, 2), None, 50, None, ValueError),
            (t.ones(1, 100, 2), None, None, 3, ValueError),
        ]:
            with self.subTest(f"dims: {dims}, nof_joints: {nof_joints}, joint_dim: {joint_dim}"):
                with self.assertRaises(exception_type):
                    validate_key_points(key_points=key_points, dims=dims, joint_dim=joint_dim, nof_joints=nof_joints)

    def test_kp_length(self):
        self.assertTrue(t.allclose(validate_key_points(DUMMY_KEY_POINTS, length=1, dims=None), DUMMY_KEY_POINTS))

    def test_kp_length_exception(self):
        with self.assertRaises(ValidationException) as e:
            _ = validate_key_points(DUMMY_KEY_POINTS, length=4, dims=None)
        self.assertTrue("Key-point length is expected to be 4 but got 1" in str(e.exception), msg=e.exception)


class TestValidateBBoxes(unittest.TestCase):

    def test_validate_bboxes(self):
        for bboxes, dims, box_format, output in [
            (DUMMY_BBOX, 2, None, DUMMY_BBOX),
            (DUMMY_BBOX, None, None, DUMMY_BBOX),
            (DUMMY_BBOX, None, tv_tensors.BoundingBoxFormat.XYWH, DUMMY_BBOX),
            (
                tv_tensors.BoundingBoxes(t.ones(4), format="XYWH", canvas_size=(10, 10)),
                2,
                None,
                tv_tensors.BoundingBoxes(t.ones((1, 4)), format="XYWH", canvas_size=(10, 10)),
            ),
        ]:
            with self.subTest(msg=f"bboxes: {bboxes}, dims: {dims}"):
                self.assertTrue(
                    t.allclose(
                        validate_bboxes(bboxes=bboxes, dims=dims, box_format=box_format),
                        output,
                    )
                )

    def test_validate_bboxes_exceptions(self):
        for bboxes, dims, box_format, exception_type in [
            (t.ones((1, 4)), 2, None, TypeError),
            (np.ones((1, 4)), 2, None, TypeError),
            (DUMMY_BBOX, None, tv_tensors.BoundingBoxFormat.XYXY, ValueError),
            (DUMMY_BBOX, None, tv_tensors.BoundingBoxFormat.CXCYWH, ValueError),
        ]:
            with self.subTest():
                with self.assertRaises(exception_type):
                    validate_bboxes(bboxes=bboxes, dims=dims, box_format=box_format),

    def test_bbox_length(self):
        self.assertTrue(t.allclose(validate_bboxes(DUMMY_BBOX, length=1, dims=None), DUMMY_BBOX))

    def test_bbox_length_exception(self):
        with self.assertRaises(ValidationException) as e:
            _ = validate_bboxes(DUMMY_BBOX, length=4, dims=None)
        self.assertTrue("Bounding box length is expected to be 4 but got 1" in str(e.exception), msg=e.exception)


class TestValidateDimensions(unittest.TestCase):

    def test_validate_dimensions(self):
        for tensor, dims, l, output in [
            (t.ones((1, 1)), 2, None, t.ones((1, 1))),
            (t.ones((1, 1)), 2, 1, t.ones((1, 1))),
            (t.ones((2, 2, 2)), 3, 2, t.ones((2, 2, 2))),
            (t.ones((1, 1, 1, 1, 5)), 1, 5, t.ones(5)),
            (t.ones(5), 5, None, t.ones((1, 1, 1, 1, 5))),
            (t.ones((2, 1, 5)), 2, 2, t.ones((2, 5))),
            (np.ones((2, 5), dtype=np.int32), 2, None, t.ones(size=(2, 5), dtype=t.int32)),
            ([[13, 7]], 2, 13, t.tensor([[13, 7]])),
        ]:
            with self.subTest(msg=f"tensor: {tensor}, dims: {dims}"):
                self.assertTrue(
                    t.allclose(
                        validate_dimensions(tensor=tensor, dims=dims),
                        output,
                    )
                )

    def test_validate_dimensions_exceptions_dims(self):
        for tensor, dims, exception_type in [
            ([[1, 1], [1]], 1, TypeError),
            ("dummy", 1, TypeError),
            (t.ones((2, 2, 5)), 2, ValueError),
            (t.ones((2, 1, 5)), 1, ValueError),
        ]:
            with self.subTest(msg=f"tensor: {tensor}, dims: {dims}, exception_type={exception_type}"):
                with self.assertRaises(exception_type):
                    validate_dimensions(tensor=tensor, dims=dims),

    def test_validate_dimensions_exceptions_length(self):
        with self.assertRaises(ValidationException) as e:
            _ = validate_dimensions(t.ones((5, 2)), dims=2, length=2)
        self.assertTrue("length is expected to be" in str(e.exception), msg=e.exception)


class TestValidateFilePaths(unittest.TestCase):

    def test_validate_file_path(self):
        full_path = tuple([os.path.normpath(os.path.join(PROJECT_ROOT, DUMMY_FP_STRING))])
        for file_path in [
            DUMMY_FP_STRING,
            os.path.join(PROJECT_ROOT, DUMMY_FP_STRING),
            DUMMY_FP,
            [DUMMY_FP_STRING],
        ]:
            with self.subTest(msg=f"file_path: {file_path}"):
                self.assertEqual(
                    validate_filepath(file_paths=file_path),
                    full_path,
                )

    def test_multiple_file_paths(self):
        full_path = os.path.normpath(os.path.join(PROJECT_ROOT, DUMMY_FP_STRING))
        resulting_paths = tuple([full_path for _ in range(B)])

        r = validate_filepath(file_paths=DUMMY_FP_BATCH)
        self.assertEqual(r, resulting_paths)

    def test_validate_file_path_with_size(self):
        path = tuple([os.path.normpath(os.path.join(PROJECT_ROOT, DUMMY_FP_STRING))])
        for file_paths, length in [
            (DUMMY_FP_STRING, 1),
            (path, 1),
            (tuple(path for _ in range(5)), 5),
        ]:
            with self.subTest(msg=f"file_path: {file_paths}, length: {length}"):
                fps = validate_filepath(file_paths, length=length)
                self.assertEqual(len(fps), length)

    def test_validate_filepath_exceptions(self):
        for fps, exception_type in [
            ("", InvalidPathException),
            (os.path.join(PROJECT_ROOT, "dummy"), InvalidPathException),
        ]:
            with self.subTest(msg=f"filepath: {fps}, exception_type={exception_type}"):
                with self.assertRaises(exception_type):
                    validate_filepath(file_paths=fps),

    def test_validate_filepath_with_wrong_size(self):
        with self.assertRaises(ValidationException) as e:
            _ = validate_filepath(
                file_paths=tuple(
                    [os.path.normpath(os.path.join(PROJECT_ROOT, "tests/test_data/images/866-200x300.jpg"))]
                ),
                length=2,
            )
        self.assertTrue("Expected 2 paths but got 1" in str(e.exception), msg=e.exception)

        with self.assertRaises(ValidationException) as e:
            _ = validate_filepath(file_paths="tests/test_data/images/866-200x300.jpg", length=2)
        self.assertTrue("Expected 2 paths but got a single path" in str(e.exception), msg=e.exception)


class TestValidateIDs(unittest.TestCase):

    def test_validate_ids(self):
        for tensor, output in [
            (1, t.ones(1).to(dtype=t.long)),
            (123456, t.tensor([123456]).long()),
            (t.ones((1, 1, 100, 1)).int(), t.ones(100).to(dtype=t.long)),
            (t.ones((2, 1)).long(), t.ones(2).to(dtype=t.long)),
            (t.ones(20).to(dtype=t.int32), t.ones(20).to(dtype=t.long)),
            (t.tensor([[1, 2, 3, 4]]), t.tensor([1, 2, 3, 4]).long()),
        ]:
            with self.subTest(msg=f"ids: {tensor}"):
                self.assertTrue(t.allclose(validate_ids(ids=tensor), output))

    def test_validate_ids_exceptions(self):
        for tensor, exception_type in [
            (np.ones((1, 10)), TypeError),
            (t.ones((2, 5)).float(), TypeError),
            (t.ones((2, 5)).int(), ValueError),
            (t.ones((1, 2, 1, 5)).int(), ValueError),
        ]:
            with self.subTest():
                with self.assertRaises(exception_type):
                    validate_ids(ids=tensor),

    def test_ids_length(self):
        self.assertTrue(t.allclose(validate_ids(DUMMY_ID, length=1), DUMMY_ID))

    def test_ids_length_exception(self):
        with self.assertRaises(ValidationException) as e:
            _ = validate_ids(DUMMY_ID, length=4)
        self.assertTrue("IDs length is expected to be 4 but got 1" in str(e.exception), msg=e.exception)


class TestValidateHeatmaps(unittest.TestCase):

    def test_validate_heatmaps(self):
        for tensor, dims, n_j, output in [
            (DUMMY_HM_TENSOR, None, None, DUMMY_HM),
            (t.zeros((J, 10, 20)), 4, J, tv_tensors.Mask(t.zeros((1, J, 10, 20)))),
            (t.zeros((1, 1, J, 10, 20)), 3, J, tv_tensors.Mask(t.zeros((J, 10, 20)))),
        ]:
            with self.subTest(msg=f"heatmap: {tensor}"):
                self.assertTrue(t.allclose(validate_heatmaps(tensor, dims=dims, nof_joints=n_j), output))

    def test_validate_heatmaps_exceptions(self):
        for tensor, n_j, exception_type in [
            (np.ones((1, 10)), None, TypeError),
            (t.ones((2, 5)), J, ValueError),
            (t.ones((J + 1, 2, 5)), J, ValueError),
        ]:
            with self.subTest(f"shape: {tensor.shape}, n_j: {n_j}, excp: {exception_type}"):
                with self.assertRaises(exception_type):
                    validate_heatmaps(tensor, nof_joints=n_j),

    def test_hm_length(self):
        self.assertTrue(t.allclose(validate_heatmaps(DUMMY_HM, length=1, dims=None), DUMMY_HM))

    def test_hm_length_exception(self):
        with self.assertRaises(ValidationException) as e:
            _ = validate_heatmaps(DUMMY_HM, length=4, dims=None)
        self.assertTrue("Heatmap length is expected to be 4 but got 1" in str(e.exception), msg=e.exception)


class TestValidateValue(unittest.TestCase):

    def test_validate_value(self):
        for value, data, validation, result in [
            (None, ..., None, True),
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
            (1, ..., float, False),
            (1.0, ..., float, True),
            (["1", "2", "3"], 1, "contains", False),
            (["1", "2", "3"], "1", "contains", True),
            (..., ..., "optional", True),
            # combined validations
            ("dummy", {"dummy": "test"}, "NodePath", True),
            ("test", {"dummy": "test"}, "NodePath", False),
            (["dummy"], {"dummy": 1}, "NodePath", True),
            (["dummy", "test"], {"dummy": {"test": 1}}, "NodePath", True),
            (["dummy", "dummy"], {"dummy": {"test": 1}}, "NodePath", False),
            ([["dummy"], ["dummy", "test"]], {"dummy": {"test": 1}}, "NodePaths", True),
            (["dummy", "dummy"], {"dummy": {"test": 1}}, "NodePaths", True),
            ([["test"]], {"dummy": {"test": 1}}, "NodePaths", False),
            # logic
            (1, int, "all", True),
            (None, "None", "all", True),
            (1, ("gt", 0), "all", True),
            (1, ("gt", 1), "all", False),
            (1, [int, ("gt", 0)], "all", True),
            (1, [int, ("gt", 1)], "all", False),
            (["a", "b"], str, "forall", True),
            (1, int, "forall", False),
            ([None], "None", "forall", True),
            ([1, 2, 3], str, "forall", False),
            ([1, 2, 3], int, "forall", True),
            ([1, 2, 3], [int], "forall", True),
            (
                [1, 2, 3],
                [int, ("gte", 1)],
                "forall",
                True,
            ),
            (
                ["a", "b", "c"],
                [str, ("in", ["a", "b"])],
                "forall",
                False,
            ),
            (
                ["a", "b"],
                [str, ("in", ["a", "b"])],
                "forall",
                True,
            ),
            ("a", "a", "eq", True),
            ("a", "b", "eq", False),
            ("a", "a", "neq", False),
            ("a", "b", "neq", True),
            (1, str, "not", True),
            (1, ("neq", 1), "not", True),
            (1, [("eq", 1), int], "not", False),
            (1, [float, ("gt", 0), ("gte", 1)], "any", True),
            (1, [int, ("gt", 1)], "any", True),
            (1, [int, ("gt", 1)], "xor", True),
            (1, [float, ("gt", 1)], "xor", False),
            (1, [float, ("gt", 0)], "xor", True),
            (1, [int, ("gt", 0)], "xor", False),
        ]:
            with self.subTest(msg=f"value: {value}, data: {data}, validation: {validation}"):
                self.assertEqual(validate_value(value, data, validation), result)

    def test_validate_value_raises(self):
        for value, data, validation, exception_type in [
            (None, None, "dummy", KeyError),
            ([1, 2, 3], "a", "shorter", ValidationException),
            # forall followed by tuple with multiple values
            ([1, 2, 3], (int, ("gt", 0)), "forall", ValidationException),
        ]:
            with self.subTest(msg=f"value: {value}, data: {data}, validation: {validation}"):
                with self.assertRaises(exception_type):
                    validate_value(value, data, validation),

    def test_nested_validations(self):
        for value, data, validation, valid in [
            ("cuda", str, "any", True),
            ("cuda", ("in", ["cuda", "cpu"]), "any", True),
            ("cuda", [("in", ["cuda", "cpu"]), ("instance", t.device)], "any", True),
            ("cpu", [("in", ["cuda", "cpu"]), ("instance", t.device)], "any", True),
            ("gpu", [("in", ["cuda", "cpu"]), ("instance", t.device)], "any", False),
            (t.device("cuda"), [("in", ["cuda", "cpu"]), ("instance", t.device)], "any", True),
            ("cuda", [("in", ["cuda", "cpu"]), ("instance", t.device)], "xor", True),
            ("cpu", [("in", ["cuda", "cpu"]), ("instance", t.device)], "xor", True),
            ("gpu", [("in", ["cuda", "cpu"]), ("instance", t.device)], "xor", False),
            (t.device("cuda"), [("in", ["cuda", "cpu"]), ("instance", t.device)], "xor", True),
            (None, "None", "all", True),
            (None, ("not", "not None"), "all", True),
            (None, ["None", ("not", "not None")], "all", True),
            (1, [("gte", 1), "not None", ("lte", 1.1), ("eq", 1), int], "all", True),
            ([1, 2, 3], list, "all", True),
            ([1, 2, 3], [list, ("forall", [int, ("gt", 0)]), ("forall", ("gt", 0)), ("forall", int)], "all", True),
        ]:
            with self.subTest(msg=f"value {value}, validation {validation}"):
                self.assertEqual(validate_value(value, data, validation), valid)


if __name__ == "__main__":
    unittest.main()
