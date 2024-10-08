import os.path
import shutil
import unittest
from math import ceil

import imagesize
import numpy as np
import torch as t
import torchvision.transforms.v2 as tvt
from torchvision import tv_tensors as tvte
from torchvision.transforms.v2.functional import crop as tvt_crop, resize as tvt_resize

from dgs.utils.exceptions import ValidationException
from dgs.utils.files import to_abspath
from dgs.utils.image import (
    combine_images_to_video,
    compute_padding,
    create_mask_from_polygons,
    CustomCropResize,
    CustomResize,
    CustomToAspect,
    CustomTransformValidator,
    load_image,
    load_image_list,
    load_video,
)
from dgs.utils.types import Image, Images, ImgShape
from dgs.utils.validation import validate_bboxes, validate_key_points
from helper import load_test_image, load_test_images_list, test_multiple_devices

# Map image name to shape.
# Shape is torch shape and therefore [C x H x W].
TEST_IMAGES: dict[str, tuple[int, int, int]] = {
    "866-200x300.jpg": (3, 300, 200),  # near quadratic higher
    "866-300x200.jpg": (3, 200, 300),  # near quadratic wider
    "866-200x800.jpg": (3, 800, 200),  # higher
    "866-800x200.jpg": (3, 200, 800),  # wide
    "866-1000x1000.jpg": (3, 1000, 1000),  # large
    "file_example_PNG_500kB.png": (3, 566, 850),  # other image
    "torch_person.jpg": (3, 640, 480),  # person
}


def create_bbox(H: int, W: int) -> tvte.BoundingBoxes:
    """Create a valid bounding box with its corners on the corners of the original image.

    Shape: ``[1 x 4]``
    """
    return validate_bboxes(
        tvte.BoundingBoxes(
            t.tensor([0, 0, W, H]),
            format=tvte.BoundingBoxFormat.XYWH,
            canvas_size=(H, W),  # H W
            dtype=t.float32,
        )
    )


def create_coordinate_diagonal(
    H: int, W: int, amount: int = 11, left: float = 0, top: float = 0, is_3d: bool = False
) -> t.Tensor:
    """Create valid key_points within the image.

    The key points form a diagonal from the point (left, top) dividing a rectangle with a given height H and width W.
    Shape: ``[1 x amount x 3 if is_3d else 2]``
    """
    step_size_w = W / (amount - 1)
    step_size_h = H / (amount - 1)
    if is_3d:
        return validate_key_points(
            t.tensor([[left + i * step_size_w, top + i * step_size_h, 0] for i in range(amount)])
        )
    return validate_key_points(t.tensor([[left + i * step_size_w, top + i * step_size_h] for i in range(amount)]))


def create_tensor_batch_data(
    image: Image,
    out_shape: ImgShape,
    mode: str,
    bbox: tvte.BoundingBoxes = None,
    key_points: t.Tensor = None,
    **kwargs,
) -> dict[str, any]:
    """Given data, create a structured data dictionary with detached clones of each tensor."""
    H, W = image.shape[-2:]

    return {
        "image": image.detach().clone(),
        "box": bbox.detach().clone() if bbox is not None else create_bbox(H, W),
        "keypoints": (
            key_points.detach().clone()
            if key_points is not None
            else create_coordinate_diagonal(H, W, is_3d=kwargs.get("is_3d", False))
        ),
        "mode": mode,
        "output_size": out_shape,
        **kwargs,
    }


def create_list_batch_data(
    images: Images,
    out_shape: ImgShape,
    mode: str,
    bbox: tvte.BoundingBoxes = None,
    key_points: t.Tensor = None,
    **kwargs,
) -> dict[str, any]:
    """Given data, create a structured data dictionary with detached clones of each tensor."""
    H, W = images[0].shape[-2:]

    return {
        "images": [img.detach().clone() for img in images],
        "box": bbox.detach().clone() if bbox is not None else create_bbox(H, W),
        "keypoints": (
            key_points.detach().clone()
            if key_points is not None
            else create_coordinate_diagonal(H, W, is_3d=kwargs.get("is_3d", False))
        ),
        "mode": mode,
        "output_size": out_shape,
        **kwargs,
    }


class TestImageUtils(unittest.TestCase):

    def test_compute_padding(self):
        for width, height, target_aspect, paddings, msg in [
            # paddings is LTRB
            (11, 11, 2, [5, 0, 6, 0], "square to landscape"),
            (11, 11, 0.5, [0, 5, 0, 6], "square to portrait"),
            (11, 11, 1, [0, 0, 0, 0], "square to square - zeros"),
            (200, 50, 1, [0, 75, 0, 75], "landscape to square"),
            (50, 75, 1, [12, 0, 13, 0], "portrait to square"),
            (200, 100, 4, [100, 0, 100, 0], "landscape to wider landscape"),
            (400, 100, 2, [0, 50, 0, 50], "landscape to slim landscape"),
            (100, 200, 1 / 4, [0, 100, 0, 100], "portrait to long portrait"),
            (50, 75, 5 / 6, [6, 0, 6, 0], "portrait to short portrait"),  # (w=62.5,h=75)
            (50, 100, 2, [75, 0, 75, 0], "portrait to landscape"),
            (150, 75, 0.5, [0, 112, 0, 113], "landscape to portrait"),
            (1920, 1080, 192 / 256, [0, 740, 0, 740], "landscape to portrait - huge values"),
            (62, 46, 256 / 192, [0, 0, 0, 0], "landscape to portrait - tiny values"),
            (46, 62, 192 / 256, [0, 0, 0, 0], "portrait to landscape - tiny values"),
            (81, 107, 0.75, [0, 0, 0, 1], "portrait to portrait - tiny values"),
            (1e8, 1e8 + 1, 1, [0, 0, 0, 0], "keep similar - huge values"),
        ]:
            with self.subTest(msg=msg):
                self.assertListEqual(compute_padding(old_w=width, old_h=height, target_aspect=target_aspect), paddings)

    def test_compute_padding_exceptions(self):
        with self.assertRaises(ValueError) as e:
            _ = compute_padding(0, 1, 1)
        self.assertTrue("Old height and width should be greater than zero" in str(e.exception), msg=e.exception)
        with self.assertRaises(ValueError) as e:
            _ = compute_padding(1, 0, 1)
        self.assertTrue("Old height and width should be greater than zero" in str(e.exception), msg=e.exception)
        with self.assertRaises(ValueError) as e:
            _ = compute_padding(1, 1, -1)
        self.assertTrue("Target aspect should be greater than zero, but is -1" in str(e.exception), msg=e.exception)
        with self.assertRaises(ValueError) as e:
            _ = compute_padding(1, 1, 1e-12)
        self.assertTrue("Target aspect should be greater than zero" in str(e.exception), msg=e.exception)
        with self.assertRaises(ValueError) as e:
            _ = compute_padding(1, 1, 0)
        self.assertTrue("Target aspect should be greater than zero, but is 0" in str(e.exception), msg=e.exception)

    @test_multiple_devices
    def test_create_mask_from_polygon(self, device: t.device):
        full = t.ones((10, 10), dtype=t.bool, device=device)
        empty = t.zeros((10, 10), dtype=t.bool, device=device)

        for shape, px, py, out in [
            ((10, 10), [[0, 9, 9, 0]], [[0, 0, 9, 9]], full),  # CW
            ((10, 10), [[0, 0, 9, 9]], [[0, 9, 9, 0]], full),  # CCW
            ((10, 10), [], [], empty),
            # lower tri
            ((10, 10), [[0, 0, 9]], [[0, 9, 9]], empty + full.tril()),
            # upper tri
            ((10, 10), [[0, 9, 9]], [[0, 9, 0]], empty + full.triu()),
            # small box in big box
            ((3, 3), [[1]], [[1]], t.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=t.bool, device=device)),
            # multiple overlapping chunks on the diagonal
            (
                (10, 10),
                [[i, i + 2, i + 2, i] for i in range(0, 9)],
                [[0, 0, 9, 9] for _ in range(0, 9)],
                full,
            ),
        ]:
            with self.subTest(msg="d: {}, shape: {}, px: {}, py: {}".format(device, shape, px, py)):
                mask = create_mask_from_polygons(shape, px, py, device=device)
                self.assertEqual(mask.device, device)
                self.assertEqual(mask.shape, t.Size(shape))

                self.assertTrue(t.allclose(mask, out), mask)

    def test_create_mask_from_polygon_raises(self):
        with self.assertRaises(ValueError) as e:
            _ = create_mask_from_polygons((1, 1), [[1, 2, 3]], [[1, 2, 3], [1, 2, 3]])
        self.assertTrue("of polygon_x 1 did not match the length of polygon_y 2." in str(e.exception), msg=e.exception)

        with self.assertRaises(ValueError) as e:
            _ = create_mask_from_polygons((1, 1), [[1]], [[1, 2]])
        self.assertTrue("1 did not match the length of the y-coordinates 2." in str(e.exception), msg=e.exception)


class TestVideo(unittest.TestCase):
    def test_load_video(self):
        for fp, shape in [
            ("./tests/test_data/videos/3209828-sd_426_240_25fps.mp4", (345, 3, 240, 426)),
        ]:
            with self.subTest(msg=f"image name: {fp}"):
                self.assertEqual(load_video(fp).shape, shape)

    def test_combine_images_to_video(self):
        for imgs, video_file, out_shape in [
            (
                tvte.Image(load_video("./tests/test_data/videos/3209828-sd_426_240_25fps.mp4", device="cpu")),
                "./tests/test_data/video_out/test1.mp4",
                (345, 3, 240, 426),
            ),
            (
                load_test_image("866-200x300.jpg", device="cpu").squeeze(0),
                "./tests/test_data/video_out/test2.mpeg",
                (1, 3, 300, 200),
            ),
            (
                load_test_images_list(["866-200x300.jpg", "866-200x300.jpg"], device="cpu"),
                "./tests/test_data/video_out/test3.mpeg",
                (2, 3, 300, 200),
            ),
        ]:
            with self.subTest(msg="video_file: {}, out_shape: {}".format(video_file, out_shape)):
                # save
                combine_images_to_video(imgs=imgs, video_file=video_file)
                # reload
                vid = load_video(video_file)
                self.assertEqual(vid.shape, out_shape)

        shutil.rmtree("./tests/test_data/video_out/")

    def test_combine_exceptions(self):
        with self.assertRaises(TypeError) as e:
            combine_images_to_video(imgs=np.ones(1), video_file="./tests/test_data/video_out/test1.mp4")
        self.assertTrue("Unknown input format. Got" in str(e.exception), msg=e.exception)


class TestImage(unittest.TestCase):

    @test_multiple_devices
    def test_load_single_image(self, device: t.device):
        for file_name, shape in TEST_IMAGES.items():
            for dtype, min_, max_ in [
                (t.float32, 0.0, 1.0),
                (t.uint8, 0, 255),
            ]:
                with self.subTest(msg=f"image name: {file_name}, dtype: {dtype}, device: {device}"):
                    fp = to_abspath(os.path.join("./tests/test_data/images/", file_name))
                    img = load_image(fp, dtype=dtype, device=device)

                    self.assertEqual(img.shape[-3:], shape)
                    self.assertEqual(imagesize.get(fp), shape[-1:-3:-1])
                    self.assertTrue(img.max().item() <= max_)
                    self.assertTrue(img.min().item() >= min_)
                    self.assertEqual(img.dtype, dtype)
                    self.assertEqual(img.device, device)

    @test_multiple_devices
    def test_load_multiple_images_resized(self, device: t.device):
        fps = tuple(to_abspath(os.path.join("./tests/test_data/images/", fn)) for fn in TEST_IMAGES)
        size: ImgShape = (300, 500)
        for dtype in [t.float32, t.uint8]:
            with self.subTest(msg=f"dtype: {dtype}, device: {device}"):
                imgs = load_image(fps, force_reshape=True, output_size=size, device=device, dtype=dtype)
                self.assertEqual(imgs.shape, t.Size((len(TEST_IMAGES), 3, 300, 500)))
                self.assertEqual(imgs.dtype, dtype)
                self.assertEqual(imgs.device, device)

    def test_load_multiple_images(self):
        for img_paths, res_shape in [
            (
                tuple("tests/test_data/images/866-200x300.jpg" for _ in range(10)),
                t.Size((10, 3, 300, 200)),
            ),
            (
                ("tests/test_data/images/866-256x256.jpg", "tests/test_data/images/866-256x256.jpg"),
                t.Size((2, 3, 256, 256)),
            ),
        ]:
            with self.subTest(msg="img_paths: {}, res_shape: {}".format(img_paths, res_shape)):
                imgs = load_image(img_paths)
                self.assertEqual(res_shape, imgs.shape)

    def test_load_multiple_images_exception(self):
        fps = tuple(to_abspath(os.path.join("./tests/test_data/images/", fn)) for fn in TEST_IMAGES)
        with self.assertRaises(ValueError):
            load_image(fps)

    @test_multiple_devices
    def test_load_image_list(self, device: t.device):
        for filepaths, dtype, hw in [
            ("tests/test_data/images/866-200x300.jpg", t.float32, [t.Size((1, 3, 300, 200))]),
            (
                tuple("tests/test_data/images/866-200x300.jpg" for _ in range(3)),
                t.float32,
                [t.Size((1, 3, 300, 200)) for _ in range(3)],
            ),
            (
                tuple(os.path.join("./tests/test_data/images/", k) for k in TEST_IMAGES),
                t.uint8,
                [t.Size((1, *v)) for v in TEST_IMAGES.values()],
            ),
        ]:
            with self.subTest(msg="filepaths: {}, dtype: {}, hw: {}".format(filepaths, dtype, hw)):
                image_list = load_image_list(filepath=filepaths, dtype=dtype, device=device)
                self.assertTrue(isinstance(image_list, list))
                self.assertEqual(len(image_list), len(hw))
                for image, shape in zip(image_list, hw):
                    self.assertTrue(isinstance(image, tvte.Image))
                    self.assertEqual(image.shape, shape)
                    self.assertEqual(image.device, device)
                    self.assertEqual(image.dtype, dtype)

    def test_load_empty_image_list(self):
        image_list = load_image_list(tuple())
        self.assertEqual(image_list, [])


class TestCustomTransformValidator(unittest.TestCase):

    def test_validate_inputs_exceptions(self):
        ctv_self = CustomTransformValidator()
        for args, n_keys, raised_exception in [
            (tuple([None]), None, TypeError),
            (tuple(), None, TypeError),
            (tuple([{"other": None}]), ["any"], KeyError),
            (tuple([{"any": None}]), ["any"], ValidationException),
        ]:
            with self.subTest(msg=f"args: {args}, necessary_keys: {n_keys}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception):
                    CustomTransformValidator._validate_inputs(ctv_self, *args, necessary_keys=n_keys)

    def test_validate_bboxes_exceptions(self):
        for data, raised_exception in [
            (t.ones(1, 4), TypeError),
            (tvte.BoundingBoxes(t.ones(1, 4), canvas_size=(10, 10), format="XYXY"), ValueError),
        ]:
            with self.subTest(msg=f"data: {data}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception):
                    CustomTransformValidator._validate_bboxes(data)

    def test_validate_key_points_exceptions(self):
        for data, raised_exception in [
            (np.ones((1, 4)), TypeError),
            (t.ones((1, 2, 3, 4)), ValueError),
            (t.ones((1, 2)), ValueError),
        ]:
            with self.subTest(msg=f"data: {data}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception):
                    CustomTransformValidator._validate_key_points(data)

    def test_validate_mode_exceptions(self):
        for data, str_dict, raised_exception in [
            ("dummy", tuple(), KeyError),
            ("fill-pad", {}, KeyError),
            ("fill-pad", {"fill": None}, KeyError),
        ]:
            with self.subTest(msg=f"data: {data}, str_dict: {str_dict}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception):
                    CustomTransformValidator._validate_mode(data, str_dict)

    def test_validate_image_exceptions(self):
        for data, raised_exception in [
            (np.ones((1, 4)), TypeError),
            (t.ones((1, 2, 3, 4)), TypeError),
        ]:
            with self.subTest(msg=f"data: {data}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception):
                    CustomTransformValidator._validate_image(data)

    def test_validate_images_exceptions(self):
        inst = CustomTransformValidator()
        for data, raised_exception, err_msg in [
            (load_test_image("866-200x300.jpg"), TypeError, "images should be a list of tv_tensors.Image"),
            (
                tuple(load_test_images_list(["866-200x300.jpg"])),
                TypeError,
                "images should be a list of tv_tensors.Image",
            ),
            ([t.ones((1, 2, 3, 4))], TypeError, "image should be a tv_tensors.Image"),
        ]:
            with self.subTest(msg=f"data: {data}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception) as e:
                    inst._validate_images(imgs=data)
                self.assertTrue(err_msg in str(e.exception), msg=e.exception)

    def test_validate_output_size_exceptions(self):
        for data, raised_exception in [
            (None, TypeError),  # None is a faulty type
            ((1, 2, 3, 4), TypeError),  # too long
            ((-1, 4), ValueError),
            ((4, -4), ValueError),
            ((1.4, 1), ValueError),
            ((1e4, 1), ValueError),
        ]:
            with self.subTest(msg=f"data: {data}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception):
                    CustomTransformValidator._validate_output_size(data)


class TestCustomToAspect(unittest.TestCase):

    def test_distort_image(self):
        out_shapes: list[ImgShape] = [(100, 100), (200, 100), (100, 200)]
        mode = "distort"
        for out_shape in out_shapes:
            distort_transform = tvt.Compose([CustomToAspect()])

            for img_name in TEST_IMAGES.keys():
                img = load_test_image(img_name)

                H, W = img.shape[-2:]

                for _3d in [False, True]:
                    with self.subTest(msg=f"img_name: {img_name}, out_shape: {out_shape}"):
                        data = create_tensor_batch_data(image=img, out_shape=out_shape, mode=mode, is_3d=_3d)

                        # get result
                        res: dict[str, any] = distort_transform(data)

                        # test result
                        new_image = res["image"]
                        new_bboxes = res["box"]
                        new_coords = res["keypoints"]

                        # test image shape - should not be modified
                        self.assertEqual(new_image.shape, img.shape)
                        # test image: should not have been changed without calling resize!
                        self.assertTrue(t.allclose(img.detach().clone(), new_image))
                        # test bbox: should not have been changed without calling resize!
                        self.assertTrue(t.allclose(create_bbox(H, W), new_bboxes))
                        # test key_points: should not have been changed without calling resize!
                        self.assertTrue(t.allclose(create_coordinate_diagonal(H, W, is_3d=_3d), new_coords))
                        # test output_size: should not have changed
                        self.assertEqual(out_shape, res["output_size"])
                        # test mode: should not have changed
                        self.assertEqual(mode, res["mode"])

    def test_distort_image_resize(self):
        out_shapes: list[ImgShape] = [(100, 100), (200, 100), (100, 200)]

        for out_shape in out_shapes:
            h, w = out_shape
            distort_resize_transform = tvt.Compose([CustomToAspect(), CustomResize()])

            for img_name in TEST_IMAGES.keys():
                img = load_test_image(img_name)
                H, W = img.shape[-2:]

                for _3d in [True, False]:
                    with self.subTest(msg=f"img_name: {img_name}, out_shape: {out_shape}"):
                        data = create_tensor_batch_data(image=img, out_shape=out_shape, mode="distort", is_3d=_3d)

                        # get result
                        res: dict[str, any] = distort_resize_transform(data)

                        # test result
                        new_image = res["image"]
                        new_bboxes = res["box"]
                        new_coords = res["keypoints"]

                        # test image shape
                        self.assertEqual(new_image.shape, t.Size((1, 3, *out_shape)))

                        # check if image is close by resizing original
                        self.assertTrue(
                            t.allclose(tvt.Resize(size=(h, w), antialias=True)(img.detach().clone()), new_image)
                        )

                        # test bbox
                        self.assertTrue(
                            t.allclose(
                                tvte.BoundingBoxes([0, 0, w, h], format="XYWH", canvas_size=(H, W), dtype=t.float32),
                                new_bboxes,
                            )
                        )
                        # test key_points
                        self.assertTrue(
                            t.allclose(
                                create_coordinate_diagonal(h, w, is_3d=_3d),
                                new_coords,
                            )
                        )
                        # test output_size: should not have changed
                        self.assertEqual(out_shape, res["output_size"])
                        # test mode: should not have changed
                        self.assertEqual("distort", res["mode"])

    def test_pad_image(self):
        out_shapes: list[ImgShape] = [(100, 100), (200, 100), (100, 200)]

        for out_shape in out_shapes:
            h, w = out_shape

            for img_name in TEST_IMAGES.keys():
                img = load_test_image(img_name)

                H, W = img.shape[-2:]

                for _3d in [True, False]:
                    for mode in [m for m in CustomToAspect.modes if m.endswith("-pad")]:
                        with self.subTest(msg=f"mode: {mode}, img_name: {img_name}, out_shape: {out_shape}"):
                            data = create_tensor_batch_data(image=img, out_shape=out_shape, mode=mode, is_3d=_3d)

                            # fill-pad mode needs additional kwarg fill
                            if mode == "fill-pad":
                                data["fill"] = 100

                            left, top, right, bottom = compute_padding(old_w=W, old_h=H, target_aspect=w / h)

                            try:
                                # get result - without Resize!
                                res: dict[str, any] = CustomToAspect()(data)
                            except ValueError as e:
                                # catch symmetric and reflect where the padding is bigger than the image
                                self.assertTrue(
                                    mode in ["symmetric-pad", "reflect-pad"]
                                    and (max(left, right) >= W or max(top, bottom) >= H)
                                )
                                self.assertEqual(
                                    "In padding modes reflect and symmetric, "
                                    "the padding can not be bigger than the image.",
                                    str(e),
                                )
                                # continue with the other tests, there is no result available!
                                continue

                            # test result
                            new_image = res["image"]
                            new_bboxes = res["box"]
                            new_coords = res["keypoints"]

                            # test image shape:
                            self.assertEqual(new_image.shape, t.Size((1, 3, H + top + bottom, W + left + right)))

                            # test image is sub-image, shape is: [B x C x H x W]
                            self.assertTrue(t.allclose(img, new_image[:, :, top : H + top, left : W + left]))
                            # test bboxes: bboxes should have shifted xy but the same w and h (without resizing)
                            self.assertTrue(
                                t.allclose(
                                    tvte.BoundingBoxes(
                                        [left, top, W, H], format="XYWH", canvas_size=(H, W), dtype=t.float32
                                    ),
                                    new_bboxes,
                                )
                            )

                            # test key points: diagonal of key_points has to stay diagonal, just shifted
                            self.assertTrue(
                                t.allclose(
                                    create_coordinate_diagonal(H, W, is_3d=_3d)
                                    + t.tensor([left, top, 0] if _3d else [left, top]),
                                    new_coords,
                                )
                            )
                            # test output_size: should not have changed
                            self.assertEqual(out_shape, res["output_size"])
                            # test mode: should not have changed
                            self.assertEqual(mode, res["mode"])

    def test_inside_crop(self):
        out_shapes: list[ImgShape] = [(100, 100), (200, 100), (100, 200)]
        mode = "inside-crop"

        for out_shape in out_shapes:
            h, w = out_shape

            for img_name in TEST_IMAGES.keys():
                img = load_test_image(img_name)

                H, W = img.shape[-2:]

                for _3d in [False, True]:
                    with self.subTest(msg=f"mode: {mode}, img_name: {img_name}, out_shape: {out_shape}, 3d: {_3d}"):
                        data = create_tensor_batch_data(image=img, out_shape=out_shape, mode=mode, is_3d=_3d)

                        nh = min(int(W / w * h), H)
                        nw = min(int(H / h * w), W)
                        left = 0.5 * (W - nw)
                        top = 0.5 * (H - nh)

                        # get result - without Resize!
                        res: dict[str, any] = CustomToAspect()(data)

                        # test result
                        new_image = res["image"]
                        new_bboxes = res["box"]
                        new_coords = res["keypoints"]

                        # test image shape: subtract padding from image shape
                        self.assertEqual(new_image.shape, t.Size((1, 3, nh, nw)))
                        # test image is sub-image, shape is: [B x C x nh x nw]
                        self.assertTrue(
                            t.allclose(
                                img[:, :, int(top) : nh + int(top), ceil(left) : nw + ceil(left)],
                                new_image,
                            )
                        )
                        # test bboxes: bboxes should have shifted xy but the same w and h (without resizing)
                        self.assertTrue(
                            t.allclose(
                                tvte.BoundingBoxes(
                                    [-left, -top, W, H], format="XYWH", canvas_size=(H, W), dtype=t.float32
                                ),
                                new_bboxes,
                            )
                        )

                        # test key points: diagonal of key_points has to stay diagonal, just shifted
                        self.assertTrue(
                            t.allclose(
                                create_coordinate_diagonal(H, W, is_3d=_3d)
                                - t.tensor([left, top, 0] if _3d else [left, top], dtype=t.float32),
                                new_coords,
                            )
                        )
                        # test output_size: should not have changed
                        self.assertEqual(out_shape, res["output_size"])
                        # test mode: should not have changed
                        self.assertEqual(mode, res["mode"])


class TestCustomResize(unittest.TestCase):

    def test_resize(self):
        out_shapes: list[ImgShape] = [(100, 100), (200, 100), (100, 200)]
        resize_transform = tvt.Compose([CustomResize()])
        for out_shape in out_shapes:
            h, w = out_shape

            for img_name in TEST_IMAGES.keys():
                for _3d in [True, False]:
                    with self.subTest(msg=f"img_name: {img_name}, out_shape: {out_shape}"):
                        img = load_test_image(img_name)

                        H, W = img.shape[-2:]

                        data = create_tensor_batch_data(image=img, out_shape=out_shape, mode="dummy", is_3d=_3d)

                        # get result
                        res: dict[str, any] = resize_transform(data)

                        # test result
                        new_image = res["image"]
                        new_bboxes = res["box"]
                        new_coords = res["keypoints"]

                        # test image shape:
                        self.assertEqual(new_image.shape, t.Size((1, 3, *out_shape)))
                        # test image: image is just resized
                        self.assertTrue(
                            t.allclose(
                                tvt_resize(img.detach().clone(), size=list(out_shape), antialias=True),
                                new_image,
                            )
                        )
                        # test bboxes: is just resized or the full resized image
                        self.assertTrue(
                            t.allclose(
                                tvt_resize(create_bbox(H, W), size=list(out_shape), antialias=True),
                                new_bboxes,
                            )
                        )
                        self.assertTrue(t.allclose(create_bbox(h, w), new_bboxes))
                        # test key points: diagonal of key_points has to stay diagonal in the new image
                        self.assertTrue(t.allclose(create_coordinate_diagonal(h, w, is_3d=_3d), new_coords))
                        # test output_size: should not have changed
                        self.assertEqual(out_shape, res["output_size"])
                        # test mode: should not have changed
                        self.assertEqual("dummy", res["mode"])


class TestCustomCropResize(unittest.TestCase):

    def test_outside_crop_single_image(self):
        out_shapes: list[ImgShape] = [(500, 500), (200, 100), (100, 200)]
        bbox_l, bbox_t, bbox_w, bbox_h = 20, 30, 50, 40
        mode = "outside-crop"

        for out_shape in out_shapes:
            h, w = out_shape
            for img_name in TEST_IMAGES.keys():
                img = load_test_image(img_name)
                H, W = img.shape[-2:]

                custom_bbox: tvte.BoundingBoxes = tvte.BoundingBoxes(
                    t.tensor([bbox_l, bbox_t, bbox_w, bbox_h]),
                    canvas_size=(H, W),
                    format=tvte.BoundingBoxFormat.XYWH,
                    dtype=t.float32,
                )

                for _3d in [True, False]:
                    custom_diag = create_coordinate_diagonal(H=bbox_h, W=bbox_w, left=bbox_l, top=bbox_t, is_3d=_3d)

                    with self.subTest(msg=f"mode: {mode}, img_name: {img_name}, out_shape: {out_shape}"):
                        data = create_list_batch_data(
                            images=[img], out_shape=out_shape, mode=mode, bbox=custom_bbox, key_points=custom_diag
                        )

                        # get result - these are the resized and stacked images!
                        res: dict[str, any] = CustomCropResize()(data)

                        new_image = res["image"]
                        new_bboxes = res["box"]
                        new_coords = res["keypoints"]

                        # test image shape:
                        self.assertEqual(new_image.shape, t.Size((1, 3, *out_shape)))
                        # test image: image is sub-image, shape is: [B x C x H x W]
                        l_pad, t_pad, r_pad, b_pad = compute_padding(old_w=bbox_w, old_h=bbox_h, target_aspect=w / h)
                        self.assertTrue(
                            t.allclose(
                                tvt_resize(
                                    img[
                                        :,  # B
                                        :,  # C
                                        bbox_t - t_pad : bbox_t + bbox_h + b_pad,  # H
                                        bbox_l - l_pad : bbox_l + bbox_w + r_pad,  # W
                                    ],
                                    size=[h, w],
                                    antialias=True,
                                ),
                                new_image,
                            )
                        )
                        # test bboxes: should stay the same
                        self.assertTrue(t.allclose(custom_bbox, new_bboxes))
                        # test key points: diagonal of key_points has to stay diagonal TODO
                        self.assertTrue(
                            t.allclose(
                                create_coordinate_diagonal(
                                    H=h - t_pad * w / bbox_w - b_pad * w / bbox_w,  # h - top - bottom
                                    W=w - l_pad * h / bbox_h - r_pad * h / bbox_h,  # w - left - right
                                    left=l_pad * h / bbox_h,
                                    top=t_pad * w / bbox_w,
                                    is_3d=_3d,
                                ),
                                new_coords,
                            )
                        )
                        # test output_size: should not have changed
                        self.assertEqual(out_shape, res["output_size"])
                        # test mode: should not have changed
                        self.assertEqual(mode, res["mode"])

    def test_single_image_3d(self):
        img = tvte.Image(load_test_image("866-200x300.jpg").squeeze(0))
        out_shape = (100, 100)
        data = create_list_batch_data(
            images=[img],
            out_shape=out_shape,
            mode="zero-pad",
            bbox=tvte.BoundingBoxes([0, 0, 300, 200], canvas_size=(200, 300), format="XYWH"),
            key_points=t.ones((1, 3, 2)),
        )
        res = CustomCropResize()(data)
        new_img = res["image"]
        self.assertTrue(isinstance(new_img, tvte.Image))
        self.assertTrue(new_img.ndim == 4)
        self.assertEqual(new_img.shape, t.Size((1, 3, *out_shape)))

    def test_other_modes_single_image(self):
        out_shape = (100, 200)
        bbox_l, bbox_t, bbox_w, bbox_h = 20, 30, 50, 40

        for mode in CustomToAspect.modes:
            if mode == "outside-crop":
                continue

            transform = tvt.Compose(
                [
                    CustomToAspect(),
                    CustomResize(),
                ]
            )

            for img_name in TEST_IMAGES.keys():
                for _3d in [True, False]:
                    with self.subTest(msg=f"mode: {mode}, img_name: {img_name}, out_shape: {out_shape}"):
                        img = load_test_image(img_name)
                        H, W = img.shape[-2:]

                        custom_bbox: tvte.BoundingBoxes = tvte.BoundingBoxes(
                            t.tensor([bbox_l, bbox_t, bbox_w, bbox_h]),
                            canvas_size=(H, W),
                            format=tvte.BoundingBoxFormat.XYWH,
                            dtype=t.float32,
                        )
                        # diagonal through box
                        custom_diag = create_coordinate_diagonal(H=bbox_h, W=bbox_w, left=bbox_l, top=bbox_t, is_3d=_3d)

                        data = create_list_batch_data(
                            images=[img],
                            out_shape=out_shape,
                            mode=mode,
                            bbox=custom_bbox,
                            key_points=custom_diag,
                            fill=0,
                        )

                        # get result - these are the resized and stacked images!
                        res: dict[str, any] = CustomCropResize()(data)

                        new_image = res["image"]
                        new_bboxes = res["box"]
                        new_coords = res["keypoints"]

                        # test image shape:
                        self.assertEqual(new_image.shape, t.Size((1, 3, *out_shape)))
                        # test image: image is sub-image, shape is: [B x C x H x W]
                        img_crop = tvte.wrap(tvt_crop(img, bbox_t, bbox_l, bbox_h, bbox_w), like=img)
                        crop_resized_img = transform(
                            create_tensor_batch_data(image=img_crop, mode=mode, out_shape=out_shape, fill=0)
                        )["image"]
                        self.assertTrue(t.allclose(crop_resized_img, new_image))
                        # test bboxes: should stay the same
                        self.assertTrue(t.allclose(custom_bbox, new_bboxes))
                        # test key points: diagonal of key_points has to stay diagonal
                        self.assertEqual(new_coords.shape, custom_diag.shape)
                        # test output_size: should not have changed
                        self.assertEqual(out_shape, res["output_size"])
                        # test mode: should not have changed
                        self.assertEqual(mode, res["mode"])

    def test_batched_input(self):
        out_shapes: list[ImgShape] = [(500, 500), (200, 100), (100, 200)]
        bbox_l, bbox_t, bbox_w, bbox_h = [20, 30, 50, 40]
        modes = ["outside-crop", "zero-pad"]
        B = 2

        for mode in modes:
            for out_shape in out_shapes:
                for img_name in TEST_IMAGES.keys():
                    imgs: Images = load_test_images_list([img_name for _ in range(B)])
                    H, W = imgs[0].shape[-2:]

                    custom_bbox: tvte.BoundingBoxes = tvte.BoundingBoxes(
                        t.tensor([[bbox_l, bbox_t, bbox_w, bbox_h], [bbox_l, bbox_t, bbox_w, bbox_h]]),
                        canvas_size=(H, W),
                        format=tvte.BoundingBoxFormat.XYWH,
                        dtype=t.float32,
                    )

                    for _3d in [True, False]:
                        custom_diag = create_coordinate_diagonal(H=bbox_h, W=bbox_w, left=bbox_l, top=bbox_t, is_3d=_3d)
                        custom_diag = custom_diag.expand(2, -1, -1)

                        with self.subTest(msg=f"mode: {mode}, img_name: {img_name}, out_shape: {out_shape}"):
                            data = create_list_batch_data(
                                images=imgs, out_shape=out_shape, mode=mode, bbox=custom_bbox, key_points=custom_diag
                            )

                            # get result - these are the resized and stacked images!
                            res: dict[str, any] = CustomCropResize()(data)

                            new_image = res["image"]
                            new_bboxes = res["box"]
                            new_coords = res["keypoints"]

                            self.assertEqual(new_image.shape, t.Size((B, 3, *out_shape)))
                            self.assertEqual(new_bboxes.shape, t.Size((B, 4)))
                            self.assertEqual(new_coords.shape, custom_diag.shape)

                            self.assertTrue(t.allclose(new_image[0], new_image[1]))
                            self.assertTrue(t.allclose(new_bboxes[0], new_bboxes[1]))
                            self.assertTrue(t.allclose(new_coords[0], new_coords[1]))

    def test_exceptions(self):
        for images, bboxes, coords, exception, err_msg in [
            (
                load_test_images_list(["866-200x300.jpg", "866-200x300.jpg"]),
                create_bbox(10, 10),  # 1 x 4
                t.zeros((2, 21, 2)),
                ValueError,
                "Expected bounding boxes 1 and key points 2 to have the same number of dimensions",
            ),
            (
                load_test_images_list(["866-200x300.jpg"]),  # just 1 image
                tvte.BoundingBoxes(t.zeros((2, 4)), canvas_size=(10, 10), format="xywh"),
                t.zeros((2, 21, 2)),
                ValueError,
                "Expected the same amount of images 1 and bounding boxes 2",
            ),
        ]:
            with self.subTest(msg=f"err_msg: {err_msg}, bboxes: {bboxes.shape}, coords: {coords.shape}"):
                data = create_list_batch_data(
                    images=images, out_shape=(100, 100), mode="zero-pad", bbox=bboxes, key_points=coords
                )

                with self.assertRaises(exception) as e:
                    _ = CustomCropResize()(data)
                self.assertTrue(err_msg in str(e.exception), msg=e.exception)


if __name__ == "__main__":
    unittest.main()
