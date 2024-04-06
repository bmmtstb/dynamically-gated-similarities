import os.path
import unittest
from math import ceil

import imagesize
import numpy as np
import torch
import torchvision.transforms.v2 as tvt
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import crop as tvt_crop, resize as tvt_resize

from dgs.utils.exceptions import ValidationException
from dgs.utils.files import to_abspath
from dgs.utils.image import (
    compute_padding,
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
    "866-200x300.jpg": (3, 300, 200),
    "866-200x800.jpg": (3, 800, 200),
    "866-300x200.jpg": (3, 200, 300),
    "866-500x500.jpg": (3, 500, 500),
    "866-500x1000.jpg": (3, 1000, 500),
    "866-800x200.jpg": (3, 200, 800),
    "866-1000x500.jpg": (3, 500, 1000),
    "866-1000x1000.jpg": (3, 1000, 1000),
    "file_example_PNG_500kB.png": (3, 566, 850),
    "torch_person.jpg": (3, 640, 480),
}


def create_bbox(H: int, W: int) -> tv_tensors.BoundingBoxes:
    """Create a valid bounding box with its corners on the corners of the original image.

    Shape: ``[1 x 4]``
    """
    return validate_bboxes(
        tv_tensors.BoundingBoxes(
            torch.tensor([0, 0, W, H]),
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=(H, W),  # H W
            dtype=torch.float32,
        )
    )


def create_coordinate_diagonal(
    H: int, W: int, amount: int = 11, left: float = 0, top: float = 0, is_3d: bool = False
) -> torch.Tensor:
    """Create valid key_points within the image.

    The key points form a diagonal from the point (left, top) dividing a rectangle with a given height H and width W.
    Shape: ``[1 x amount x 3 if is_3d else 2]``
    """
    step_size_w = W / (amount - 1)
    step_size_h = H / (amount - 1)
    if is_3d:
        return validate_key_points(
            torch.tensor([[left + i * step_size_w, top + i * step_size_h, 0] for i in range(amount)])
        )
    return validate_key_points(torch.tensor([[left + i * step_size_w, top + i * step_size_h] for i in range(amount)]))


def create_tensor_batch_data(
    image: Image,
    out_shape: ImgShape,
    mode: str,
    bbox: tv_tensors.BoundingBoxes = None,
    key_points: torch.Tensor = None,
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
    bbox: tv_tensors.BoundingBoxes = None,
    key_points: torch.Tensor = None,
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
        ]:
            with self.subTest(msg=msg):
                self.assertListEqual(compute_padding(old_w=width, old_h=height, target_aspect=target_aspect), paddings)


class TestVideo(unittest.TestCase):
    def test_load_video(self):
        for fp, shape in [
            ("./tests/test_data/images/3209828-sd_426_240_25fps.mp4", (345, 3, 240, 426)),
        ]:
            with self.subTest(msg=f"image name: {fp}"):
                self.assertEqual(load_video(fp).shape, shape)


class TestImage(unittest.TestCase):

    @test_multiple_devices
    def test_load_single_image(self, device: torch.device):
        for file_name, shape in TEST_IMAGES.items():
            for dtype, min_, max_ in [
                (torch.float32, 0.0, 1.0),
                (torch.uint8, 0, 255),
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
    def test_load_multiple_images_resized(self, device: torch.device):
        fps = tuple(to_abspath(os.path.join("./tests/test_data/images/", fn)) for fn in TEST_IMAGES)
        size: ImgShape = (300, 500)
        for dtype in [torch.float32, torch.uint8]:
            with self.subTest(msg=f"dtype: {dtype}, device: {device}"):
                imgs = load_image(fps, force_reshape=True, output_size=size, device=device, dtype=dtype)
                self.assertEqual(imgs.shape, torch.Size((len(TEST_IMAGES), 3, 300, 500)))
                self.assertEqual(imgs.dtype, dtype)
                self.assertEqual(imgs.device, device)

    def test_load_multiple_images(self):
        for img_paths, res_shape in [
            (
                tuple("tests/test_data/images/866-200x300.jpg" for _ in range(10)),
                torch.Size((10, 3, 300, 200)),
            ),
            (
                ("tests/test_data/images/866-256x256.jpg", "tests/test_data/images/866-256x256.jpg"),
                torch.Size((2, 3, 256, 256)),
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
    def test_load_image_list(self, device: torch.device):
        for filepaths, dtype, hw in [
            ("tests/test_data/images/866-200x300.jpg", torch.float32, [torch.Size((1, 3, 300, 200))]),
            (
                tuple("tests/test_data/images/866-200x300.jpg" for _ in range(3)),
                torch.float32,
                [torch.Size((1, 3, 300, 200)) for _ in range(3)],
            ),
            (
                tuple(os.path.join("./tests/test_data/images/", k) for k in TEST_IMAGES),
                torch.uint8,
                [torch.Size((1, *v)) for v in TEST_IMAGES.values()],
            ),
        ]:
            with self.subTest(msg="filepaths: {}, dtype: {}, hw: {}".format(filepaths, dtype, hw)):
                image_list = load_image_list(filepath=filepaths, dtype=dtype, device=device)
                self.assertTrue(isinstance(image_list, list))
                self.assertEqual(len(image_list), len(hw))
                for image, shape in zip(image_list, hw):
                    self.assertTrue(isinstance(image, tv_tensors.Image))
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
            (torch.ones(1, 4), TypeError),
            (tv_tensors.BoundingBoxes(torch.ones(1, 4), canvas_size=(10, 10), format="XYXY"), ValueError),
        ]:
            with self.subTest(msg=f"data: {data}, raised_exception: {raised_exception}"):
                with self.assertRaises(raised_exception):
                    CustomTransformValidator._validate_bboxes(data)

    def test_validate_key_points_exceptions(self):
        for data, raised_exception in [
            (np.ones((1, 4)), TypeError),
            (torch.ones((1, 2, 3, 4)), ValueError),
            (torch.ones((1, 2)), ValueError),
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
            (torch.ones((1, 2, 3, 4)), TypeError),
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
            ([torch.ones((1, 2, 3, 4))], TypeError, "image should be a tv_tensors.Image"),
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
                        self.assertEqual(new_image.shape[-2:], img.shape[-2:])
                        # test image: should not have been changed without calling resize!
                        self.assertTrue(torch.allclose(img.detach().clone(), new_image))
                        # test bbox: should not have been changed without calling resize!
                        self.assertTrue(torch.allclose(create_bbox(H, W), new_bboxes))
                        # test key_points: should not have been changed without calling resize!
                        self.assertTrue(torch.allclose(create_coordinate_diagonal(H, W, is_3d=_3d), new_coords))
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
                        self.assertEqual(new_image.shape[-2:], out_shape)

                        # check if image is close by resizing original
                        self.assertTrue(
                            torch.allclose(tvt.Resize(size=(h, w), antialias=True)(img.detach().clone()), new_image)
                        )

                        # test bbox
                        self.assertTrue(
                            torch.allclose(
                                tv_tensors.BoundingBoxes(
                                    [0, 0, w, h], format="XYWH", canvas_size=(H, W), dtype=torch.float32
                                ),
                                new_bboxes,
                            )
                        )
                        # test key_points
                        self.assertTrue(
                            torch.allclose(
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

                            l, t, r, b = compute_padding(old_w=W, old_h=H, target_aspect=w / h)

                            try:
                                # get result - without Resize!
                                res: dict[str, any] = CustomToAspect()(data)
                            except ValueError as e:
                                # catch symmetric and reflect where the padding is bigger than the image
                                self.assertTrue(
                                    mode in ["symmetric-pad", "reflect-pad"] and (max(l, r) >= W or max(t, b) >= H)
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
                            self.assertEqual(new_image.shape[-2:], (H + t + b, W + l + r))

                            # test image is sub-image, shape is: [B x C x H x W]
                            self.assertTrue(torch.allclose(img, new_image[:, :, t : H + t, l : W + l]))
                            # test bboxes: bboxes should have shifted xy but the same w and h (without resizing)
                            self.assertTrue(
                                torch.allclose(
                                    tv_tensors.BoundingBoxes(
                                        [l, t, W, H], format="XYWH", canvas_size=(H, W), dtype=torch.float32
                                    ),
                                    new_bboxes,
                                )
                            )

                            # test key points: diagonal of key_points has to stay diagonal, just shifted
                            self.assertTrue(
                                torch.allclose(
                                    create_coordinate_diagonal(H, W, is_3d=_3d)
                                    + torch.tensor([l, t, 0] if _3d else [l, t]),
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
                        l = 0.5 * (W - nw)
                        t = 0.5 * (H - nh)

                        # get result - without Resize!
                        res: dict[str, any] = CustomToAspect()(data)

                        # test result
                        new_image = res["image"]
                        new_bboxes = res["box"]
                        new_coords = res["keypoints"]

                        # test image shape: subtract padding from image shape
                        self.assertEqual(new_image.shape[-2:], (nh, nw))
                        # test image is sub-image, shape is: [B x C x nh x nw]
                        self.assertTrue(
                            torch.allclose(
                                img[:, :, int(t) : nh + int(t), ceil(l) : nw + ceil(l)],
                                new_image,
                            )
                        )
                        # test bboxes: bboxes should have shifted xy but the same w and h (without resizing)
                        self.assertTrue(
                            torch.allclose(
                                tv_tensors.BoundingBoxes(
                                    [-l, -t, W, H], format="XYWH", canvas_size=(H, W), dtype=torch.float32
                                ),
                                new_bboxes,
                            )
                        )

                        # test key points: diagonal of key_points has to stay diagonal, just shifted
                        self.assertTrue(
                            torch.allclose(
                                create_coordinate_diagonal(H, W, is_3d=_3d)
                                - torch.tensor([l, t, 0] if _3d else [l, t], dtype=torch.float32),
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
                        self.assertEqual(new_image.shape[-2:], out_shape)
                        # test image: image is just resized
                        self.assertTrue(
                            torch.allclose(
                                tvt_resize(img.detach().clone(), size=list(out_shape), antialias=True),
                                new_image,
                            )
                        )
                        # test bboxes: is just resized or the full resized image
                        self.assertTrue(
                            torch.allclose(
                                tvt_resize(create_bbox(H, W), size=list(out_shape), antialias=True),
                                new_bboxes,
                            )
                        )
                        self.assertTrue(torch.allclose(create_bbox(h, w), new_bboxes))
                        # test key points: diagonal of key_points has to stay diagonal in the new image
                        self.assertTrue(torch.allclose(create_coordinate_diagonal(h, w, is_3d=_3d), new_coords))
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

                custom_bbox: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
                    torch.tensor([bbox_l, bbox_t, bbox_w, bbox_h]),
                    canvas_size=(H, W),
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    dtype=torch.float32,
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
                        self.assertEqual(new_image.shape[-2:], out_shape)
                        # test image: image is sub-image, shape is: [B x C x H x W]
                        l_pad, t_pad, r_pad, b_pad = compute_padding(old_w=bbox_w, old_h=bbox_h, target_aspect=w / h)
                        self.assertTrue(
                            torch.allclose(
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
                        self.assertTrue(torch.allclose(custom_bbox, new_bboxes))
                        # test key points: diagonal of key_points has to stay diagonal TODO
                        self.assertTrue(
                            torch.allclose(
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
        img = tv_tensors.Image(load_test_image("866-200x300.jpg").squeeze(0))
        data = create_list_batch_data(
            images=[img],
            out_shape=(100, 100),
            mode="zero-pad",
            bbox=tv_tensors.BoundingBoxes([0, 0, 300, 200], canvas_size=(200, 300), format="XYWH"),
            key_points=torch.ones((1, 3, 2)),
        )
        res = CustomCropResize()(data)
        self.assertTrue(isinstance(res["image"], tv_tensors.Image))
        self.assertTrue(res["image"].ndim == 4)

    def test_other_modes_single_image(self):
        out_shape = (100, 200)
        h, w = out_shape
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

                        custom_bbox: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
                            torch.tensor([bbox_l, bbox_t, bbox_w, bbox_h]),
                            canvas_size=(H, W),
                            format=tv_tensors.BoundingBoxFormat.XYWH,
                            dtype=torch.float32,
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
                        self.assertEqual(new_image.shape[-2:], out_shape)
                        # test image: image is sub-image, shape is: [B x C x H x W]
                        img_crop = tv_tensors.wrap(tvt_crop(img, bbox_t, bbox_l, bbox_h, bbox_w), like=img)
                        crop_resized_img = transform(
                            create_tensor_batch_data(image=img_crop, mode=mode, out_shape=out_shape, fill=0)
                        )["image"]
                        self.assertTrue(torch.allclose(crop_resized_img, new_image))
                        # test bboxes: should stay the same
                        self.assertTrue(torch.allclose(custom_bbox, new_bboxes))
                        # test key points: diagonal of key_points has to stay diagonal
                        self.assertEqual(new_coords.shape, custom_diag.shape)
                        # test output_size: should not have changed
                        self.assertEqual(out_shape, res["output_size"])
                        # test mode: should not have changed
                        self.assertEqual(mode, res["mode"])

    def test_outside_crop_batched_input(self):
        out_shapes: list[ImgShape] = [(500, 500), (200, 100), (100, 200)]
        bbox_l, bbox_t, bbox_w, bbox_h = [20, 30, 50, 40]
        mode = "outside-crop"

        for out_shape in out_shapes:
            for img_name in TEST_IMAGES.keys():
                imgs: Images = load_test_images_list([img_name, img_name])
                H, W = imgs[0].shape[-2:]

                custom_bbox: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
                    torch.tensor([[bbox_l, bbox_t, bbox_w, bbox_h], [bbox_l, bbox_t, bbox_w, bbox_h]]),
                    canvas_size=(H, W),
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    dtype=torch.float32,
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

                        self.assertEqual(tuple(new_image.shape[-2:]), out_shape)
                        self.assertEqual(new_image.shape[0], 2)
                        self.assertEqual(new_bboxes.shape[0], 2)
                        self.assertEqual(new_coords.shape[0], 2)

                        self.assertTrue(torch.allclose(new_image[0], new_image[1]))
                        self.assertTrue(torch.allclose(new_bboxes[0], new_bboxes[1]))
                        self.assertTrue(torch.allclose(new_coords[0], new_coords[1]))

    def test_exceptions(self):
        for images, bboxes, coords, exception, err_msg in [
            (
                load_test_images_list(["866-200x300.jpg", "866-200x300.jpg"]),
                create_bbox(10, 10),  # 1 x 4
                torch.zeros((2, 21, 2)),
                ValueError,
                "Expected bounding boxes 1 and key points 2 to have the same number of dimensions",
            ),
            (
                load_test_images_list(["866-200x300.jpg"]),  # just 1 image
                tv_tensors.BoundingBoxes(torch.zeros((2, 4)), canvas_size=(10, 10), format="xywh"),
                torch.zeros((2, 21, 2)),
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
