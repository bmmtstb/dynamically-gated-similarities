import os.path
import unittest

import imagesize
import torch
import torchvision.transforms.v2 as tvt
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import resize as tvt_resize

from dgs.utils.files import project_to_abspath
from dgs.utils.image import compute_padding, CustomCropResize, CustomResize, CustomToAspect, load_image, load_video
from dgs.utils.types import ImgShape, TVImage
from dgs.utils.validation import validate_bboxes, validate_key_points
from helper import load_test_image

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
}


def create_bbox(H: int, W: int) -> tv_tensors.BoundingBoxes:
    """Create a valid bounding box with its corners on the corners of the original image."""
    return validate_bboxes(
        tv_tensors.BoundingBoxes(
            torch.Tensor([0, 0, W, H]),
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=(H, W),  # H W
            dtype=torch.float32,
        )
    )


def create_coordinate_diagonal(H: int, W: int, amount: int = 11, left: float = 0, top: float = 0) -> torch.Tensor:
    """Create valid key_points within the image.

    The key points form a diagonal from the point (left, top) dividing a rectangle with a given height H and width W.
    """
    step_size_w = W / (amount - 1)
    step_size_h = H / (amount - 1)

    return validate_key_points(torch.Tensor([[left + i * step_size_w, top + i * step_size_h] for i in range(amount)]))


def create_structured_data(
    image: TVImage,
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
        "keypoints": key_points.detach().clone() if key_points is not None else create_coordinate_diagonal(H, W),
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


class TestImage(unittest.TestCase):
    def test_load_image(self):
        for file_name, shape in TEST_IMAGES.items():
            with self.subTest(msg=f"image name: {file_name}"):
                fp = project_to_abspath(os.path.join("./tests/test_data/", file_name))
                self.assertEqual(load_image(fp).shape, shape)
                self.assertEqual(imagesize.get(fp), shape[-1:-3:-1])


class TestVideo(unittest.TestCase):
    def test_load_video(self):
        for fp, shape in [
            ("./tests/test_data/file_example_AVI_640_800kB.avi", (901, 3, 360, 640)),
        ]:
            with self.subTest(msg=f"image name: {fp}"):
                self.assertEqual(load_video(fp).shape, shape)


class TestCustomTransformValidator(unittest.TestCase):
    def test_validate(self):
        # fixme
        pass


class TestCustomToAspect(unittest.TestCase):
    def test_distort_image(self):
        out_shapes: list[ImgShape] = [(100, 100), (200, 100), (100, 200)]
        mode = "distort"
        for out_shape in out_shapes:
            distort_transform = tvt.Compose([CustomToAspect()])

            for img_name in TEST_IMAGES.keys():
                with self.subTest(msg=f"img_name: {img_name}, out_shape: {out_shape}"):
                    img = load_test_image(img_name)

                    H, W = img.shape[-2:]

                    data = create_structured_data(image=img, out_shape=out_shape, mode=mode)

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
                    self.assertTrue(torch.allclose(create_coordinate_diagonal(H, W), new_coords))
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
                with self.subTest(msg=f"img_name: {img_name}, out_shape: {out_shape}"):
                    img = load_test_image(img_name)

                    H, W = img.shape[-2:]

                    data = create_structured_data(image=img, out_shape=out_shape, mode="distort")

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
                            create_coordinate_diagonal(h, w),
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

                for mode in [m for m in CustomToAspect.modes if m.endswith("-pad")]:
                    with self.subTest(msg=f"mode: {mode}, img_name: {img_name}, out_shape: {out_shape}"):
                        data = create_structured_data(image=img, out_shape=out_shape, mode=mode)

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
                                "In padding modes reflect and symmetric, the padding can not be bigger than the image.",
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
                            torch.allclose(create_coordinate_diagonal(H, W) + torch.Tensor([l, t]), new_coords)
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
                with self.subTest(msg=f"img_name: {img_name}, out_shape: {out_shape}"):
                    img = load_test_image(img_name)

                    H, W = img.shape[-2:]

                    data = create_structured_data(image=img, out_shape=out_shape, mode="dummy")

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
                    self.assertTrue(torch.allclose(create_coordinate_diagonal(h, w), new_coords))
                    # test output_size: should not have changed
                    self.assertEqual(out_shape, res["output_size"])
                    # test mode: should not have changed
                    self.assertEqual("dummy", res["mode"])


class TestCustomCropResize(unittest.TestCase):
    def test_outside_crop(self):
        out_shapes: list[ImgShape] = [(100, 100), (200, 100), (100, 200)]
        bbox_l, bbox_t, bbox_w, bbox_h = 20, 30, 50, 40

        for out_shape in out_shapes:
            h, w = out_shape
            for img_name in TEST_IMAGES.keys():
                img = load_test_image(img_name)
                H, W = img.shape[-2:]

                custom_bbox: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
                    torch.Tensor([bbox_l, bbox_t, bbox_w, bbox_h]),
                    canvas_size=(H, W),
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    dtype=torch.float32,
                )

                custom_diag = create_coordinate_diagonal(H=bbox_h, W=bbox_w, left=bbox_l, top=bbox_t)

                mode = "outside-crop"

                with self.subTest(msg=f"mode: {mode}, img_name: {img_name}, out_shape: {out_shape}"):
                    data = create_structured_data(
                        image=img, out_shape=out_shape, mode=mode, bbox=custom_bbox, key_points=custom_diag
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
                            ),
                            new_coords,
                        )
                    )
                    # test output_size: should not have changed
                    self.assertEqual(out_shape, res["output_size"])
                    # test mode: should not have changed
                    self.assertEqual(mode, res["mode"])


if __name__ == "__main__":
    unittest.main()
