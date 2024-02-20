import os
import unittest
from copy import deepcopy

import torch
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.image import load_image
from dgs.utils.states import DataSample, get_ds_data_getter
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

DUMMY_HM_TENSOR: torch.Tensor = torch.distributions.uniform.Uniform(0, 1).sample(torch.Size((1, J, 10, 20))).float()
DUMMY_HM: tv_tensors.Mask = tv_tensors.Mask(DUMMY_HM_TENSOR, dtype=torch.float32)

DUMMY_JOINT_WEIGHT: torch.Tensor = torch.tensor([i / J for i in range(J)]).view((1, J, 1)).float()

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

    def test_init_with_kwargs(self):
        fp = (os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"),)
        bbox = DUMMY_BBOX
        kp = DUMMY_KEY_POINTS
        ds = DataSample(filepath=fp, bbox=bbox, keypoints=kp, validate=False, dummy="dummy")

        self.assertEqual(ds["dummy"], "dummy")

    def test_crop_path(self):
        ds = DataSample(
            filepath=("dummy",), bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS, validate=False, crop_path=("dummy",)
        )
        self.assertEqual(ds.crop_path, ("dummy",))

    def test_len(self):
        for fps, length in [
            ((os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"),), 1),
            (
                (
                    os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"),
                    os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"),
                ),
                2,
            ),
        ]:
            with self.subTest(msg="fps: {}, length: {}".format(fps, length)):
                ds = DataSample(filepath=fps, bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS, validate=False)
                self.assertEqual(len(ds), length)

    @test_multiple_devices
    def test_to(self, device: torch.device):
        fp = (os.path.join(PROJECT_ROOT, "./tests/test_data/866-200x300.jpg"),)
        bbox = DUMMY_BBOX.cpu()
        kp = DUMMY_KEY_POINTS.cpu()
        ds = DataSample(filepath=fp, bbox=bbox, keypoints=kp, validate=False)
        ds.to(device=device)
        self.assertEqual(ds.bbox.device, device)
        self.assertEqual(ds.keypoints.device, device)

    def test_get_original_image(self):
        fp = tuple([DUMMY_FILE_PATH])
        img = load_image(filepath=fp)

        with self.assertRaises(KeyError):
            _ = DataSample(filepath=fp, bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS).image
            self.fail()

        ds2 = DataSample(filepath=fp, bbox=DUMMY_BBOX, keypoints=DUMMY_KEY_POINTS, image=img)

        self.assertTrue(torch.allclose(ds2.image, img))

    @test_multiple_devices
    def test_init_with_device(self, device: Device):
        out_id = torch.tensor(10).long().to(device=device)
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

    def test_cast_joint_weight(self):
        for weights, decimals, dtype, result in [
            (
                DUMMY_JOINT_WEIGHT.detach().clone(),
                0,
                torch.int32,
                torch.round(torch.tensor([i / J for i in range(J)]).view((1, J, 1)), decimals=0).to(dtype=torch.int32),
            ),
            (
                DUMMY_JOINT_WEIGHT.detach().clone(),
                1,
                torch.float32,
                torch.round(torch.tensor([i / J for i in range(J)]).view((1, J, 1)), decimals=1),
            ),
            (  # typecast only
                DUMMY_JOINT_WEIGHT.detach().clone(),
                -1,
                torch.int32,
                torch.tensor([i / J for i in range(J)]).view((1, J, 1)).to(dtype=torch.int32),
            ),
            (  # start with int
                DUMMY_JOINT_WEIGHT.detach().clone().int(),
                0,
                torch.int32,
                torch.round(
                    torch.tensor([i / J for i in range(J)], dtype=torch.int32).view((1, J, 1)).float(), decimals=0
                ).to(dtype=torch.int32),
            ),
        ]:
            for overwrite in [True, False]:
                with self.subTest(
                    msg="decimals: {}, dtype: {}, overwrite: {}, weights-type: {}, result: {}".format(
                        decimals, dtype, overwrite, weights.dtype, result
                    )
                ):
                    data = DUMMY_DATA.copy()
                    data["joint_weight"] = weights
                    ds = DataSample(**data)
                    dsi = deepcopy(ds)
                    r = dsi.cast_joint_weight(dtype=dtype, decimals=decimals, overwrite=overwrite)
                    self.assertEqual(r.dtype, dtype)
                    self.assertTrue(torch.allclose(r, result))
                    if overwrite:
                        self.assertTrue(torch.allclose(r, dsi.joint_weight))
                    else:
                        self.assertTrue(torch.allclose(dsi.joint_weight, ds.joint_weight))


class TestDataGetter(unittest.TestCase):
    def test_get_ds_data_getter(self):
        getter = get_ds_data_getter(["image"])
        self.assertTrue(callable(getter))
        ds = DataSample(**DUMMY_DATA.copy())
        self.assertTrue(torch.allclose(getter(ds)[0], ds.image))


if __name__ == "__main__":
    unittest.main()
