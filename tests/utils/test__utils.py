import os
import shutil
import unittest

import numpy as np
import torch
from torchvision import tv_tensors as tvte
from torchvision.tv_tensors import BoundingBoxes

from dgs.utils.config import DEF_VAL
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import is_project_file
from dgs.utils.image import CustomCropResize
from dgs.utils.types import Device
from dgs.utils.utils import (
    extract_crops_and_save,
    extract_crops_from_images,
    HidePrint,
    ids_to_one_hot,
    replace_file_type,
    torch_to_numpy,
)
from helper import capture_stdout, load_test_image, load_test_images, load_test_images_list, test_multiple_devices


class TestUtils(unittest.TestCase):
    @test_multiple_devices
    def test_torch_to_np(self, device: Device):
        for torch_tensor, numpy_array in [
            (torch.ones((5, 2), device=device, dtype=torch.int32), np.ones((5, 2), dtype=np.int32)),
        ]:
            with self.subTest(msg=f"torch_tensor: {torch_tensor}, numpy_array: {numpy_array}"):
                self.assertTrue(np.array_equal(torch_to_numpy(torch_tensor), numpy_array))

    def test_replace_image_type(self):
        for new_file_type, fp, old_types, result in [
            (".pt", "test.jpg", [".jpg", ".jpeg"], "test.pt"),
            ("pt", "test.jpg", [".jpg", ".jpeg"], "test.pt"),
            ("pt", "test.jpg", ["jpg", "jpeg"], "test.pt"),
            ("pt", "test.jpg", None, "test.pt"),
            ("_glob.pt", "test.jpg", None, "test_glob.pt"),
        ]:
            with self.subTest(
                msg="new_file_type: {}, fp: {}, old_types: {}, result: {}".format(new_file_type, fp, old_types, result)
            ):
                self.assertEqual(replace_file_type(fp=fp, new_type=new_file_type, old_types=old_types), result)

    def test_replace_image_type_exceptions(self):
        with self.assertRaises(ValueError) as e:
            _ = replace_file_type(fp="test.gif", new_type="dummy", old_types=["jpg", "jpeg"])
        self.assertTrue("Expected file type '.gif' to be in" in str(e.exception), msg=e.exception)

    def test_extract_crops_from_images(self):
        dummy_kp = torch.ones((1, 3, 2))
        dummy_box = BoundingBoxes(torch.tensor([0, 0, 200, 300]), format="XYWH", canvas_size=(300, 200))

        r = CustomCropResize()(
            {
                "images": load_test_images_list(["866-200x300.jpg"]),
                "box": dummy_box.detach().clone(),
                "keypoints": dummy_kp.detach().clone(),
                "output_size": DEF_VAL["images"]["crop_size"],
                "mode": DEF_VAL["images"]["crop_mode"],
            }
        )

        example_crop = r["image"]
        example_kp = r["keypoints"]

        quad = CustomCropResize()(
            {
                "images": load_test_images_list(["866-500x500.jpg", "866-500x500.jpg"]),
                "box": BoundingBoxes(
                    torch.tensor([[0, 0, 500, 500]]).repeat(2, 1), format="XYWH", canvas_size=(500, 500)
                ),
                "keypoints": torch.ones((2, 3, 2)),
                "output_size": DEF_VAL["images"]["crop_size"],
                "mode": DEF_VAL["images"]["crop_mode"],
            }
        )["image"]

        for imgs, bboxes, kps, res_imgs, res_kps, kwargs in [
            (
                load_test_images_list(["866-200x300.jpg"]),
                BoundingBoxes(torch.tensor([0, 0, 200, 300]), format="XYWH", canvas_size=(300, 200)),
                dummy_kp,
                example_crop,
                example_kp,
                {},
            ),
            (
                load_test_images_list(["866-200x300.jpg"]),
                BoundingBoxes(torch.tensor([0, 0, 200, 300]), format="XYWH", canvas_size=(300, 200)),
                dummy_kp,
                load_test_image("866-200x300.jpg"),
                dummy_kp,
                {"crop_size": (300, 200)},  # h w
            ),
            (
                load_test_images_list(["866-200x300.jpg", "866-200x300.jpg"]),
                BoundingBoxes(torch.tensor([[0, 0, 200, 300]]).repeat(2, 1), format="XYWH", canvas_size=(300, 200)),
                torch.ones(2, 3, 2),
                load_test_images(["866-200x300.jpg", "866-200x300.jpg"]),
                torch.ones(2, 3, 2),
                {"crop_size": (300, 200)},
            ),
            (
                load_test_images_list(["866-500x500.jpg", "866-500x500.jpg"]),
                BoundingBoxes(torch.tensor([[0, 0, 500, 500]]).repeat(2, 1), format="XYWH", canvas_size=(500, 500)),
                None,
                quad,
                None,
                {},
            ),
        ]:
            with self.subTest(
                msg="B: {}, out_size: {}, bboxes: {}, kps: {}".format(len(imgs), res_imgs.shape[-2:], bboxes, kps)
            ):
                crops, loc_kps = extract_crops_from_images(imgs=imgs, bboxes=bboxes, kps=kps, **kwargs)

                self.assertEqual(
                    crops.shape,
                    torch.Size(
                        (
                            len(imgs),
                            3,
                            *(kwargs["crop_size"] if "crop_size" in kwargs else DEF_VAL["images"]["crop_size"]),
                        )
                    ),
                )
                self.assertTrue(isinstance(crops, tvte.Image))
                self.assertTrue(torch.allclose(crops, res_imgs))

                if loc_kps is None:
                    self.assertTrue(res_kps is None)
                else:
                    self.assertTrue(torch.allclose(loc_kps, res_kps))

        self.assertTrue(torch.allclose(dummy_kp, torch.ones((1, 3, 2))), "the key points were modified")

    def test_extract_from_empty_images(self):
        imgs = []
        kps = torch.empty((0, 1, 2))
        box = BoundingBoxes(torch.tensor([0, 0, 200, 300]), format="XYWH", canvas_size=(300, 200))
        crop, kps = extract_crops_from_images(imgs=imgs, kps=kps, bboxes=box)
        self.assertEqual(kps, None)
        self.assertTrue(torch.allclose(crop, tvte.Image(torch.empty(0, 3, 1, 1))))

    def test_extract_crops_exceptions(self):
        imgs = load_test_images_list(["866-500x500.jpg"])  # 1
        box = BoundingBoxes(torch.ones((2, 4)), format="XYWH", canvas_size=(300, 200))
        with self.assertRaises(ValueError) as e:
            _ = extract_crops_from_images(imgs=imgs, bboxes=box)
        self.assertTrue(
            "Expected length of imgs 1 and number of bounding boxes 2 to match." in str(e.exception), msg=e.exception
        )

    @test_multiple_devices
    def test_extract_crops_and_save(self, device: Device):
        base_path = os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/"))
        img_src = os.path.normpath(os.path.join(base_path, "./test_data/images/"))
        crop_target = os.path.normpath(os.path.join(base_path, "./test_crops"))
        for img_shapes, crop_fps, bbox_format, kp, kwargs in [
            ([(300, 200)], ["1.jpg"], "xywh", None, {"device": device}),
            ([(300, 200), (300, 200)], ["2_1.jpg", "2_2.jpg"], "xyxy", None, {"device": device}),
            ([(300, 200) for _ in range(10)], [f"3_{i}.jpg" for i in range(10)], "xyxy", None, {"device": device}),
            (
                [(500, 1000)],
                ["4.jpg"],
                "xywh",
                None,
                {"crop_mode": "mean-pad", "crop_size": (128, 128), "quality": 50},
            ),
            ([(300, 200)], ["5.jpg"], "xywh", torch.ones((1, 11, 2)), {"device": device}),
            ([(300, 200), (300, 200)], ["6_1.jpg", "6_2.jpg"], "xyxy", torch.ones((2, 11, 2)), {"device": device}),
        ]:
            with self.subTest(
                msg=f"shapes: {img_shapes}, crop_fps: {crop_fps}, format: {bbox_format}, kp: {kp}, kwargs: {kwargs}"
            ):
                box_coords = torch.stack([torch.tensor([0, 1, 10, 21]) for _ in range(len(img_shapes))])
                bboxes = BoundingBoxes(box_coords, canvas_size=max(img_shapes), format=bbox_format)
                img_fps = [os.path.join(img_src, f"866-{shape[1]}x{shape[0]}.jpg") for shape in img_shapes]
                crop_fps = [os.path.join(crop_target, c) for c in crop_fps]
                crops, new_kp = extract_crops_and_save(
                    img_fps=img_fps, new_fps=crop_fps, boxes=bboxes, key_points=kp, **kwargs
                )

                self.assertEqual(len(crops), len(img_shapes))
                if kp is None:
                    self.assertTrue(new_kp is None)
                else:
                    self.assertEqual(len(new_kp), len(img_shapes))
                for fp in crop_fps:
                    self.assertTrue(is_project_file(fp))
                    self.assertEqual(is_project_file(replace_file_type(str(fp), new_type=".pt")), kp is not None)
        # delete crops folder in the end
        shutil.rmtree(crop_target)

    @test_multiple_devices
    def test_extract_crops_and_save_exceptions(self, device: Device):
        bbox = BoundingBoxes(torch.tensor([1, 2, 3, 4]), canvas_size=(100, 100), format="xywh")
        for img_fps, crop_fps, kp, bboxes in [
            (["dummy"], [], None, bbox),
            ([], ["dummy"], None, bbox),
            (["dummy"], ["dummy"], None, BoundingBoxes(torch.ones((2, 4)), canvas_size=(100, 100), format="xywh")),
            (["dummy"], [], torch.ones((1, 11, 2)), bbox),
            ([], ["dummy"], torch.ones((1, 11, 2)), bbox),
            (["dummy"], ["dummy"], torch.ones((2, 11, 2)), bbox),
            (
                ["dummy"],
                ["dummy"],
                torch.ones((1, 11, 2)),
                BoundingBoxes(torch.ones((2, 4)), canvas_size=(100, 100), format="xywh"),
            ),
        ]:
            with self.subTest(msg=f"img_fps, crop_fps, bbox"):
                with self.assertRaises(ValueError):
                    extract_crops_and_save(
                        img_fps=img_fps, new_fps=crop_fps, boxes=bboxes, key_points=kp, device=device
                    )

    def test_HidePrint(self):
        def printing(text: str) -> None:
            print(text)

        def no_printing(text: str) -> None:
            with HidePrint():
                print(text)

        with capture_stdout(printing, "Printed!") as output:
            self.assertTrue(output.startswith("Printed!"))

        with capture_stdout(no_printing, "Not shown!") as output:
            self.assertEqual("", output)

    def test_to_one_hot(self):
        nof_C = 10
        t = torch.arange(nof_C)
        expected = torch.diag(torch.ones(10, dtype=torch.long))
        r = ids_to_one_hot(ids=t, nof_classes=nof_C)
        self.assertTrue(torch.allclose(r, expected))


if __name__ == "__main__":
    unittest.main()
