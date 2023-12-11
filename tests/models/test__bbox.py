import unittest
from copy import deepcopy

import torch

from dgs.models.torchbbox.bbox import BoundingBox
from dgs.utils.exceptions import BoundingBoxException
from dgs.utils.types import Device
from tests.helper import test_multiple_devices

XYXY: torch.Tensor = torch.FloatTensor([1, 2, 101, 202])
XYAH: torch.Tensor = torch.FloatTensor([1, 2, 0.5, 200])
XYWH: torch.Tensor = torch.FloatTensor([1, 2, 100, 200])
YOLO: torch.Tensor = torch.FloatTensor([51, 102, 100, 200])

XYXY_BBOX: BoundingBox = BoundingBox(xyxy=deepcopy(XYXY))
XYAH_BBOX: BoundingBox = BoundingBox(xyah=deepcopy(XYAH))
XYWH_BBOX: BoundingBox = BoundingBox(xywh=deepcopy(XYWH))
YOLO_BBOX: BoundingBox = BoundingBox(yolo=deepcopy(YOLO))
ALL_BBOX: BoundingBox = BoundingBox(xyxy=XYXY, xyah=XYAH, xywh=XYWH, yolo=YOLO)


class TestBoundingBox(unittest.TestCase):
    def test_successful_creation(self):
        for kwargs, msg in [
            ({"xyxy": XYXY}, "xyxy"),
            ({"xywh": XYWH}, "xywh"),
            ({"xyah": XYAH}, "xyah"),
            ({"yolo": YOLO}, "yolo"),
            ({"xyxy": XYXY, "xywh": XYWH, "xyah": XYAH, "yolo": YOLO}, "all"),
        ]:
            with self.subTest(msg=msg):
                BoundingBox(**deepcopy(kwargs))

    def test_failed_creation(self):
        for kwargs, error_msg, msg in [
            ({}, "format has to be specified", "empty kwargs"),
            ({"wrong": XYWH}, "Unknown bbox format during initialization.", "wrong kwarg name"),
            (
                {"xyxy": XYXY, "wrong": XYWH},
                "Unknown bbox format during initialization.",
                "right and wrong kwarg name",
            ),
            ({"xyah": None}, "At least one bbox has to have a value.", "None in kwargs"),
            ({"xyxy": torch.Tensor([1, 1, 101, 201, 1])}, "have to have length of 4.", "Tensor too long"),
            ({"xyah": torch.Tensor([1, 1, 101])}, "have to have length of 4.", "Tensor too short"),
            (
                {"xyxy": XYXY, "xyah": XYXY},
                f"Noticed while converting value from xyxy to xyah",
                "Validation on two values",
            ),
            (
                {"xyxy": XYXY, "xywh": XYXY, "xyah": XYAH, "yolo": YOLO},
                "Multiple values were given, but they do not have the same value",
                "Validation on multiple values",
            ),
            ({"xyxy": torch.Tensor([101, 1, 1, 201])}, "left greater than than right", "x1>x2"),
            ({"xyxy": torch.Tensor([1, 1, 1, 201])}, "left greater than than right", "x1==x2"),
            ({"xyxy": torch.Tensor([1, 101, 101, 1])}, "top greater than than bottom", "y1>y2"),
            ({"xyxy": torch.Tensor([1, 1, 101, 1])}, "top greater than than bottom", "y1==y2"),
            ({"xywh": torch.Tensor([1, 1, 100, -1])}, "Cannot create bounding box with height of 0", "h<0"),
            ({"xywh": torch.Tensor([1, 1, -1, 100])}, "Cannot create bounding box with width of 0", "w<0"),
            ({"xywh": torch.Tensor([1, 1, 100, 0])}, "Cannot create bounding box with height of 0", "h==0"),
            ({"xywh": torch.Tensor([1, 1, 0, 100])}, "Cannot create bounding box with width of 0", "w==0"),
            ({"xyah": torch.Tensor([1, 1, 100, -1])}, "Cannot create bounding box with height of 0", "h<0"),
            ({"xyah": torch.Tensor([1, 1, -1, 100])}, "bounding box with aspect ratio smaller than 0", "a<0"),
            ({"xyah": torch.Tensor([1, 1, 1, 0])}, "Cannot create bounding box with height of 0", "h==0"),
            ({"xyah": torch.Tensor([1, 1, 0, 100])}, " bounding box with aspect ratio smaller than 0", "a==0"),
            ({"yolo": torch.Tensor([1, 1, 100, -1])}, "Cannot create bounding box with height of 0", "h<0"),
            ({"yolo": torch.Tensor([1, 1, -1, 100])}, "Cannot create bounding box with width of 0", "w<0"),
            ({"yolo": torch.Tensor([1, 1, 100, 0])}, "Cannot create bounding box with height of 0", "h==0"),
            ({"yolo": torch.Tensor([1, 1, 0, 100])}, "Cannot create bounding box with width of 0", "w==0"),
        ]:
            with self.subTest(msg=msg):
                with self.assertRaises(BoundingBoxException) as context:
                    BoundingBox(**deepcopy(kwargs))
                self.assertTrue(error_msg in str(context.exception))

    def test_creation_on_devices(self):
        if torch.cuda.is_available():
            with self.assertRaises(BoundingBoxException) as context:
                BoundingBox(
                    xyxy=torch.Tensor([1, 2, 101, 202]).to(device="cpu"),
                    xyah=torch.Tensor([1, 2, 2, 100]).to(device="cuda"),
                )
            self.assertTrue("but they are not on the same device" in str(context.exception))

    def test_successful_conversion(self):
        for bbox, msg in [
            (deepcopy(XYXY_BBOX), "xyxy"),
            (deepcopy(XYWH_BBOX), "xywh"),
            (deepcopy(XYAH_BBOX), "xyah"),
            (deepcopy(YOLO_BBOX), "yolo"),
            (ALL_BBOX, "all"),
        ]:
            with self.subTest(msg=msg):
                self.assertTrue(torch.allclose(bbox.xyxy, deepcopy(XYXY)), f"during {msg}: get xyxy")
                self.assertTrue(torch.allclose(bbox.xywh, deepcopy(XYWH)), f"during {msg}: get xywh")
                self.assertTrue(torch.allclose(bbox.xyah, deepcopy(XYAH)), f"during {msg}: get xyah")
                self.assertTrue(torch.allclose(bbox.yolo, deepcopy(YOLO)), f"during {msg}: get yolo")

    @test_multiple_devices
    def test_computed_values(self, device: Device):
        for bbox, bbox_name in [
            (deepcopy(XYXY_BBOX), "xyxy"),
            (deepcopy(XYWH_BBOX), "xywh"),
            (deepcopy(XYAH_BBOX), "xyah"),
            (deepcopy(YOLO_BBOX), "yolo"),
            (ALL_BBOX, "all"),
        ]:
            bbox = deepcopy(bbox)
            for prop_name, solution in [
                ("left", torch.tensor(1, device=device, dtype=torch.float32)),
                ("right", torch.tensor(101, device=device, dtype=torch.float32)),
                ("top", torch.tensor(2, device=device, dtype=torch.float32)),
                ("bottom", torch.tensor(202, device=device, dtype=torch.float32)),
                ("height", torch.tensor(200, device=device, dtype=torch.float32)),
                ("width", torch.tensor(100, device=device, dtype=torch.float32)),
                ("center", torch.FloatTensor([51, 102], device=device)),
                ("aspect_ratio", torch.tensor(0.5, device=device, dtype=torch.float32)),
            ]:
                with self.subTest(msg=f"{bbox_name} - {prop_name}"):
                    self.assertTrue(torch.allclose(getattr(bbox, prop_name), solution))

    @test_multiple_devices
    def test_corners(self, device: Device):
        for bbox, W, H, corners, msg in [
            (
                deepcopy(BoundingBox(xyxy=XYXY.to(device=device))),
                1000,
                1000,
                torch.IntTensor([1, 2, 101, 202], device=device),
                "Big canvas",
            ),
            (
                deepcopy(BoundingBox(xyxy=XYXY.to(device=device))),
                1000,
                100,
                torch.IntTensor([1, 2, 101, 99], device=device),
                "Small height",
            ),
            (
                deepcopy(BoundingBox(xyxy=XYXY.to(device=device))),
                100,
                1000,
                torch.IntTensor([1, 2, 99, 202], device=device),
                "Small width",
            ),
            (
                deepcopy(BoundingBox(xyxy=XYXY.to(device=device))),
                100,
                100,
                torch.IntTensor([1, 2, 99, 99], device=device),
                "Small h and w",
            ),
            (
                BoundingBox(xyxy=torch.FloatTensor([-1, -10, 10, 1], device=device)),
                1000,
                1000,
                torch.IntTensor([0, 0, 10, 1], device=device),
                "Negative bboxes coordinates",
            ),
        ]:
            with self.subTest(msg=f"{msg}"):
                self.assertTrue(torch.allclose(bbox.corners((W, H)), corners))

    @test_multiple_devices
    def test_contains(self, device: Device):
        for bbox, bbox_name in [
            (BoundingBox(xyxy=deepcopy(XYXY).to(device=device)), "xyxy"),
            (BoundingBox(xywh=deepcopy(XYWH).to(device=device)), "xywh"),
            (BoundingBox(xyah=deepcopy(XYAH).to(device=device)), "xyah"),
            (BoundingBox(yolo=deepcopy(YOLO).to(device=device)), "yolo"),
        ]:
            for point, contains in [
                (torch.Tensor([0, 0]).to(device=device, dtype=torch.float32), False),
                (torch.Tensor([1, 2]).to(device=device, dtype=torch.float32), True),
                (torch.Tensor([0.999, 1.999]).to(device=device, dtype=torch.float32), False),
                (torch.Tensor([1.0000, 2.0000]).to(device=device, dtype=torch.float32), True),
                (torch.Tensor([50, 0]).to(device=device, dtype=torch.float32), False),
                (torch.Tensor([0, 100]).to(device=device, dtype=torch.float32), False),
                (torch.Tensor([-10, -10]).to(device=device, dtype=torch.float32), False),
                (torch.Tensor([50, 202.001]).to(device=device, dtype=torch.float32), False),
                (torch.Tensor([50, 201.999]).to(device=device, dtype=torch.float32), True),
                (torch.Tensor([200, 200]).to(device=device, dtype=torch.float32), False),
                (torch.Tensor([51, 102]).to(device=device, dtype=torch.float32), True),
            ]:
                with self.subTest(msg=f"p: {point}"):
                    self.assertEqual(bbox.contains(point), contains)


if __name__ == "__main__":
    unittest.main()
