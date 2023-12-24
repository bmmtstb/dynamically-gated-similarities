import os
import unittest

import torch
from torchvision import tv_tensors

from dgs.models.states import DataSample
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.image import load_image
from dgs.utils.types import Device, FilePath, FilePaths
from tests.helper import load_test_image, test_multiple_devices

J = 20
J_dim = 2
IMG_NAME = "866-200x300.jpg"
DUMMY_IMG = load_test_image(IMG_NAME)

DUMMY_KEY_POINTS_TENSOR: torch.Tensor = torch.rand((1, J, J_dim))
DUMMY_KEY_POINTS = DUMMY_KEY_POINTS_TENSOR.detach().clone()
DUMMY_KP_BATCH: torch.Tensor = torch.cat(
    [DUMMY_KEY_POINTS.clone().detach()] + [torch.rand((1, J, J_dim)) for _ in range(9)]
)

DUMMY_BBOX_TENSOR: torch.Tensor = torch.ones((1, 4)) * 10
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)
DUMMY_BBOXES: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    torch.cat([DUMMY_BBOX_TENSOR for _ in range(10)]),
    format=tv_tensors.BoundingBoxFormat.XYWH,
    canvas_size=(1000, 1000),
)

DUMMY_FILE_PATH: FilePath = os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/test_data/" + IMG_NAME))
DUMMY_FP_BATCH: FilePaths = tuple(DUMMY_FILE_PATH for _ in range(10))

DUMMY_HM_TENSOR: torch.FloatTensor = (
    torch.distributions.uniform.Uniform(0, 1).sample(torch.Size((1, J, 10, 20))).float()
)
DUMMY_HM: tv_tensors.Mask = tv_tensors.Mask(DUMMY_HM_TENSOR, dtype=torch.float32)

DUMMY_JOINT_WEIGHT: torch.FloatTensor = torch.tensor([i / J for i in range(J)]).view((1, J, 1)).float()

DUMMY_DATA: dict[str, any] = {
    "filepath": DUMMY_FILE_PATH,
    "bbox": DUMMY_BBOX,
    "keypoints": DUMMY_KEY_POINTS_TENSOR,
    "keypoints_local": DUMMY_KEY_POINTS_TENSOR,
    "heatmap": DUMMY_HM_TENSOR,
    "image": load_test_image(IMG_NAME),
    "image_crop": DUMMY_IMG,
    "person_id": 10,
    "joint_weight": DUMMY_JOINT_WEIGHT,
}


class TestDataSample(unittest.TestCase):
    def test_init_regular(self):
        for fp, bbox, kp, out_fp, out_bbox, out_kp, validate in [
            (
                os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"),
                DUMMY_BBOX,
                DUMMY_KEY_POINTS,
                tuple([DUMMY_FILE_PATH]),
                DUMMY_BBOX,
                DUMMY_KEY_POINTS,
                True,
            ),
            (
                "./tests/test_data/866-200x300.jpg",
                DUMMY_BBOX,
                DUMMY_KEY_POINTS_TENSOR,
                tuple([DUMMY_FILE_PATH]),
                DUMMY_BBOX,
                DUMMY_KEY_POINTS,
                True,
            ),
            (  # no validation
                "./tests/test_data/866-200x300.jpg",
                DUMMY_BBOX_TENSOR,
                DUMMY_KEY_POINTS_TENSOR,
                "./tests/test_data/866-200x300.jpg",
                DUMMY_BBOX_TENSOR,
                DUMMY_KEY_POINTS_TENSOR,
                False,
            ),
            (  # batched init
                [DUMMY_FILE_PATH for _ in range(10)],
                DUMMY_BBOXES,
                DUMMY_KP_BATCH,
                DUMMY_FP_BATCH,
                DUMMY_BBOXES,
                DUMMY_KP_BATCH,
                True,
            ),
        ]:
            with self.subTest(msg=f"fp: {fp}, bbox: {bbox}, kp: {kp}, v: {validate}"):
                ds = DataSample(filepath=fp, bbox=bbox, keypoints=kp, validate=validate)
                self.assertEqual(ds.filepath, out_fp)
                self.assertTrue(torch.allclose(ds.bbox, out_bbox))
                self.assertTrue(torch.allclose(ds.keypoints, out_kp))
                self.assertEqual(ds.J, J)
                self.assertEqual(ds.joint_dim, J_dim)

    def test_get_original_image(self):
        fp = tuple([DUMMY_FILE_PATH])
        img = load_image(filepath=fp)

        with self.assertRaises(KeyError):
            _ = DataSample(filepath=fp, bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS).image
            self.fail()

        ds2 = DataSample(filepath=fp, bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS, image=img)

        self.assertTrue(torch.allclose(ds2.image, img))

    def test_cast_joint_weight(self):
        pass

    @test_multiple_devices
    def test_init_with_device(self, device: Device):
        out_id = torch.tensor(10).int().to(device=device)
        out_image = DUMMY_IMG.to(device=device)
        out_imgcrop = DUMMY_IMG.to(device=device)
        out_kp = DUMMY_KEY_POINTS.to(device=device)
        out_loc_kp = DUMMY_KEY_POINTS.to(device=device)
        out_bbox = DUMMY_BBOX.to(device=device)
        out_joint_weight = DUMMY_JOINT_WEIGHT.to(device=device)

        # set the device as kwarg vs get it from bbox
        for set_dev_init in [True, False]:
            for validate in [True, False]:  # check whether it depends on validation
                with self.subTest(msg=f"init: {set_dev_init}, validate: {validate}, device: {device}"):
                    out_fp = tuple([DUMMY_FILE_PATH]) if validate else DUMMY_FILE_PATH
                    # input data
                    data_dict = DUMMY_DATA.copy()
                    if set_dev_init:
                        data_dict["device"] = device
                    else:
                        data_dict["bbox"] = data_dict["bbox"].to(device=device)
                        data_dict["keypoints"] = data_dict["keypoints"].to(device=device)
                    data_dict["validate"] = validate

                    ds: DataSample = DataSample(**data_dict)
                    self.assertEqual(ds.device, torch.device(device))
                    self.assertTrue(torch.allclose(ds.person_id, out_id))
                    self.assertTrue(torch.allclose(ds.image, out_image))
                    self.assertTrue(torch.allclose(ds.image_crop, out_imgcrop))
                    self.assertTrue(torch.allclose(ds.keypoints, out_kp))
                    self.assertTrue(torch.allclose(ds.keypoints_local, out_loc_kp))
                    self.assertTrue(torch.allclose(ds.bbox, out_bbox))
                    self.assertTrue(torch.allclose(ds.joint_weight, out_joint_weight))
                    self.assertEqual(ds.filepath, out_fp)
                    self.assertEqual(ds.J, J)
                    self.assertEqual(ds.joint_dim, J_dim)


if __name__ == "__main__":
    unittest.main()
