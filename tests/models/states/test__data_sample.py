import os
import unittest
from copy import deepcopy

import torch
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.states import DataSample, get_ds_data_getter
from dgs.utils.types import Device, FilePaths
from tests.helper import load_test_image, test_multiple_devices

J = 20
J_dim = 2
B = 10
PID = 13

IMG_NAME = "866-200x300.jpg"
DUMMY_IMG = load_test_image(IMG_NAME)

DUMMY_KP_TENSOR: torch.Tensor = torch.rand((1, J, J_dim))
DUMMY_KP = DUMMY_KP_TENSOR.detach().clone()
DUMMY_KP_BATCH: torch.Tensor = torch.cat(
    [DUMMY_KP.clone().detach()] + [torch.rand((1, J, J_dim)) for _ in range(B - 1)]
)

DUMMY_BBOX_TENSOR: torch.Tensor = torch.ones((1, 4))
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)
DUMMY_BBOXES: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    torch.cat([DUMMY_BBOX_TENSOR for _ in range(B)]),
    format=tv_tensors.BoundingBoxFormat.XYWH,
    canvas_size=(1000, 1000),
)

DUMMY_FP_STRING: str = f"./tests/test_data/{IMG_NAME}"
DUMMY_FP: FilePaths = (os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/test_data/" + IMG_NAME)),)
DUMMY_FP_BATCH: FilePaths = tuple(os.path.normpath(os.path.join(PROJECT_ROOT, DUMMY_FP_STRING)) for _ in range(B))

DUMMY_HM_TENSOR: torch.Tensor = torch.distributions.uniform.Uniform(0, 1).sample(torch.Size((1, J, B, 20))).float()
DUMMY_HM: tv_tensors.Mask = tv_tensors.Mask(DUMMY_HM_TENSOR, dtype=torch.float32)

DUMMY_WEIGHT: torch.Tensor = torch.tensor([i / J for i in range(J)]).view((1, J, 1)).float()

DUMMY_DATA: dict[str, any] = {
    "filepath": DUMMY_FP,
    "bbox": DUMMY_BBOX,
    "keypoints": DUMMY_KP_TENSOR,
    "keypoints_local": DUMMY_KP_TENSOR,
    "heatmap": DUMMY_HM_TENSOR,
    "image": load_test_image(IMG_NAME),
    "image_crop": DUMMY_IMG,
    "person_id": PID,
    "joint_weight": DUMMY_WEIGHT,
}


class TestDataSample(unittest.TestCase):

    def test_init_regular(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):
                ds = DataSample(bbox=DUMMY_BBOX, validate=validate)
                self.assertTrue(torch.allclose(ds.bbox, DUMMY_BBOX))
                self.assertEqual(len(ds), 1)
                self.assertEqual(ds.B, 1)

                multi_ds = DataSample(bbox=DUMMY_BBOXES, validate=validate)
                self.assertTrue(torch.allclose(multi_ds.bbox, DUMMY_BBOXES))
                self.assertEqual(len(multi_ds), B)
                self.assertEqual(multi_ds.B, B)

    def test_init_with_unknown_kwarg(self):
        ds = DataSample(bbox=DUMMY_BBOX, dummy="dummy")
        self.assertEqual(ds["dummy"], "dummy")

    def test_init_with_multiple_values(self):
        for fp, bbox, kp, out_fp, out_bbox, out_kp, validate in [
            ((DUMMY_FP_STRING,), DUMMY_BBOX, DUMMY_KP_TENSOR, DUMMY_FP, DUMMY_BBOX, DUMMY_KP, True),
            (  # no validation - string will still be cast to tuple but not validated
                DUMMY_FP_STRING,
                DUMMY_BBOX_TENSOR,
                DUMMY_KP_TENSOR,
                (DUMMY_FP_STRING,),
                DUMMY_BBOX_TENSOR,
                DUMMY_KP_TENSOR,
                False,
            ),
            (
                tuple([DUMMY_FP_STRING for _ in range(B)]),
                DUMMY_BBOXES,
                DUMMY_KP_BATCH,
                DUMMY_FP_BATCH,
                DUMMY_BBOXES,
                DUMMY_KP_BATCH,
                True,
            ),
            (DUMMY_FP_BATCH, DUMMY_BBOXES, DUMMY_KP_BATCH, DUMMY_FP_BATCH, DUMMY_BBOXES, DUMMY_KP_BATCH, True),
        ]:
            with self.subTest(msg=f"v: {validate}, len: {len(bbox)} fp: {fp}, bbox: {bbox}, kp: {kp}"):
                ds = DataSample(filepath=fp, bbox=bbox, keypoints=kp, validate=validate)
                self.assertEqual(ds.filepath, out_fp)
                self.assertTrue(torch.allclose(ds.bbox, out_bbox))
                self.assertTrue(torch.allclose(ds.keypoints, out_kp))
                self.assertEqual(ds.J, J)
                self.assertEqual(ds.joint_dim, J_dim)

    def test_args_raises(self):
        with self.assertRaises(NotImplementedError) as e:
            _ = DataSample("dummy", bbox=DUMMY_BBOX)
        self.assertTrue("Unknown arguments" in str(e.exception), msg=e.exception)

    def test_J(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):

                faulty_ds = DataSample(bbox=DUMMY_BBOX, validate=validate)
                with self.assertRaises(NotImplementedError) as e:
                    _ = faulty_ds.J
                self.assertTrue("no global or local key-points in this object" in str(e.exception), msg=e.exception)

            for scope in ["keypoints", "keypoints_local"]:
                with self.subTest(msg="scope: {}, validate: {}".format(scope, validate)):

                    ds = DataSample(bbox=DUMMY_BBOX, keypoints=DUMMY_KP, validate=validate)
                    self.assertEqual(ds.J, J)

                    multi_ds = DataSample(bbox=DUMMY_BBOXES, keypoints=DUMMY_KP_BATCH, validate=validate)
                    self.assertEqual(multi_ds.J, J)

    def test_joint_dim(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):

                faulty_ds = DataSample(bbox=DUMMY_BBOX, validate=validate)
                with self.assertRaises(NotImplementedError) as e:
                    _ = faulty_ds.joint_dim
                self.assertTrue("no global or local key-points in this object" in str(e.exception), msg=e.exception)

            for scope in ["keypoints", "keypoints_local"]:
                with self.subTest(msg="scope: {}, validate: {}".format(scope, validate)):

                    ds = DataSample(**{"bbox": DUMMY_BBOX, scope: DUMMY_KP, "validate": validate})
                    self.assertEqual(ds.joint_dim, J_dim)

                    multi_ds = DataSample(**{"bbox": DUMMY_BBOXES, scope: DUMMY_KP_BATCH, "validate": validate})
                    self.assertEqual(multi_ds.joint_dim, J_dim)

    def test_keypoints(self):
        scopes = ["keypoints", "keypoints_local"]
        for validate in [True, False]:
            for i, scope in enumerate(scopes):
                with self.subTest(msg="scope: {}, validate: {}".format(scope, validate)):
                    ds = DataSample(**{"bbox": DUMMY_BBOX, scope: DUMMY_KP, "validate": validate})
                    setattr(ds, scopes[(i + 1) % 2], DUMMY_KP)

    def test_setting_bbox_fails(self):
        ds = DataSample(**DUMMY_DATA)
        with self.assertRaises(NotImplementedError) as e:
            ds.bbox = DUMMY_BBOX
        self.assertTrue("not allowed to change the bounding box of an already" in str(e.exception), msg=e.exception)

    def test_filepath_exceptions(self):
        # tuple
        with self.assertRaises(ValueError) as e:
            _ = DataSample(bbox=DUMMY_BBOX, filepath=DUMMY_FP + DUMMY_FP)
        self.assertTrue(
            "filepath must have the same number of entries as bounding-boxes. Got 2, expected 1" in str(e.exception),
            msg=e.exception,
        )
        # string
        with self.assertRaises(ValueError) as e:
            _ = DataSample(bbox=DUMMY_BBOXES, filepath=DUMMY_FP_STRING)
        self.assertTrue(f"Got a single path, expected {B}" in str(e.exception), msg=e.exception)

    def test_get_filepath_fails_as_string(self):
        ds = DataSample(bbox=DUMMY_BBOX)
        ds.data["filepath"] = DUMMY_FP_STRING
        with self.assertRaises(AssertionError) as e:
            _ = ds.filepath
        self.assertTrue("filepath must be a tuple but got" in str(e.exception), msg=e.exception)

    def test_class_id(self):
        ds = DataSample(filepath=("dummy",), bbox=DUMMY_BBOX, keypoints=DUMMY_KP, validate=False, class_id=1)
        self.assertEqual(ds.class_id, torch.ones(1, dtype=torch.long))

        ds = DataSample(
            filepath=("dummy",), bbox=DUMMY_BBOX, keypoints=DUMMY_KP, validate=False, class_id=torch.ones(1)
        )
        self.assertEqual(ds.class_id, torch.ones(1, dtype=torch.long))

    def test_crop_path(self):
        ds = DataSample(filepath=("dummy",), bbox=DUMMY_BBOX, keypoints=DUMMY_KP, validate=False, crop_path=("dummy",))
        self.assertEqual(ds.crop_path, ("dummy",))

    def test_len(self):

        for fps, length in [
            (os.path.join(PROJECT_ROOT, DUMMY_FP_STRING), 1),
            ((os.path.join(PROJECT_ROOT, DUMMY_FP_STRING),), 1),
        ]:
            with self.subTest(msg="fps: {}, length: {}".format(fps, length)):
                ds = DataSample(filepath=fps, bbox=DUMMY_BBOX, keypoints=DUMMY_KP, validate=False)
                self.assertEqual(len(ds), length)

        multi_ds = DataSample(bbox=DUMMY_BBOXES, filepath=DUMMY_FP_BATCH)
        self.assertEqual(len(multi_ds), B)

    @test_multiple_devices
    def test_to(self, device: torch.device):
        fp = DUMMY_FP
        bbox = tv_tensors.wrap(DUMMY_BBOX.cpu(), like=DUMMY_BBOX)
        kp = DUMMY_KP.cpu()
        ds = DataSample(filepath=fp, bbox=bbox, keypoints=kp, validate=False)
        ds.to(device=device)
        self.assertEqual(ds.bbox.device, device)
        self.assertEqual(ds.keypoints.device, device)

    @test_multiple_devices
    def test_init_with_device(self, device: Device):
        out_id = torch.tensor(PID).long().to(device=device)
        out_image = DUMMY_IMG.to(device=device)
        out_imgcrop = DUMMY_IMG.to(device=device)
        out_kp = DUMMY_KP.to(device=device)
        out_loc_kp = DUMMY_KP.to(device=device)
        out_bbox = DUMMY_BBOX.to(device=device)
        out_joint_weight = DUMMY_WEIGHT.to(device=device)

        # set the device as kwarg vs get it from bbox
        for set_dev_init in [True, False]:
            for validate in [True, False]:  # check whether it depends on validation
                with self.subTest(msg=f"init: {set_dev_init}, validate: {validate}, device: {device}"):
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
                    self.assertEqual(ds.J, J)
                    self.assertEqual(ds.joint_dim, J_dim)

    def test_cast_joint_weight(self):
        for weights, decimals, dtype, result in [
            (
                DUMMY_WEIGHT.detach().clone(),
                0,
                torch.int32,
                torch.round(torch.tensor([i / J for i in range(J)]).view((1, J, 1)), decimals=0).to(dtype=torch.int32),
            ),
            (
                DUMMY_WEIGHT.detach().clone(),
                1,
                torch.float32,
                torch.round(torch.tensor([i / J for i in range(J)]).view((1, J, 1)), decimals=1),
            ),
            (  # typecast only
                DUMMY_WEIGHT.detach().clone(),
                -1,
                torch.int32,
                torch.tensor([i / J for i in range(J)]).view((1, J, 1)).to(dtype=torch.int32),
            ),
            (  # start with int
                DUMMY_WEIGHT.detach().clone().int(),
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

    def test_load_image(self):
        orig_img = load_test_image(IMG_NAME)
        ds = DataSample(filepath=(DUMMY_FP,), bbox=DUMMY_BBOX, validate=False)
        multi_ds = DataSample(filepath=DUMMY_FP_BATCH, bbox=DUMMY_BBOXES, validate=False)
        no_fps = DataSample(bbox=DUMMY_BBOX, validate=False)
        img_ds = DataSample(bbox=DUMMY_BBOX, image=orig_img.clone(), validate=False)

        # get using data -> fails if not present yet
        for obj in [ds, multi_ds, no_fps]:
            with self.subTest(msg="obj: {}".format(obj)):
                with self.assertRaises(KeyError) as e:
                    _ = obj.copy().data["image"]
                self.assertTrue("'image'" in str(e.exception), msg=e.exception)
        # -> succeed if present
        orig_img = img_ds.data["image"]
        self.assertTrue(torch.allclose(orig_img, orig_img.clone()))

        # call load_image
        ds.load_image()
        img = ds.data["image"]
        self.assertTrue(isinstance(img, tv_tensors.Image))
        self.assertEqual(img.shape, orig_img.shape)
        self.assertTrue(torch.allclose(img, orig_img))

        imgs = multi_ds.load_image()
        self.assertTrue(isinstance(imgs, tv_tensors.Image))
        self.assertEqual(list(imgs.shape), [B] + list(orig_img.shape)[1:])
        self.assertTrue(torch.allclose(imgs, orig_img.repeat_interleave(B, dim=0)))

        self.assertTrue(torch.allclose(img_ds.load_image(), orig_img))

        # calling load image fails if the filepaths are not given
        with self.assertRaises(AttributeError) as e:
            _ = no_fps.load_image()
        self.assertTrue("Could not load images without proper filepaths given" in str(e.exception), msg=e.exception)

    def test_load_image_crop(self):
        orig_img = load_test_image(IMG_NAME)
        ds = DataSample(bbox=DUMMY_BBOX, crop_path=(DUMMY_FP,), validate=False)
        multi_ds = DataSample(bbox=DUMMY_BBOXES, crop_path=DUMMY_FP_BATCH, validate=False)
        no_fps = DataSample(bbox=DUMMY_BBOX, validate=False)
        img_ds = DataSample(bbox=DUMMY_BBOX, image_crop=orig_img.clone(), validate=False)

        # get using data -> fails if not present yet
        for obj in [ds, multi_ds, no_fps]:
            with self.subTest(msg="obj: {}".format(obj)):
                with self.assertRaises(KeyError) as e:
                    _ = obj.copy().data["image_crop"]
                self.assertTrue("'image_crop'" in str(e.exception), msg=e.exception)
        # -> succeed if present
        orig_img = img_ds.data["image_crop"]
        self.assertTrue(torch.allclose(orig_img, orig_img))

        # call load_image_crop
        ds.load_image_crop()
        crop = ds.data["image_crop"]
        self.assertTrue(isinstance(crop, tv_tensors.Image))
        self.assertEqual(crop.shape, orig_img.shape)
        self.assertTrue(torch.allclose(crop, orig_img))

        imgs = multi_ds.load_image_crop()
        self.assertTrue(isinstance(imgs, tv_tensors.Image))
        self.assertEqual(list(imgs.shape), [B] + list(orig_img.shape)[1:])
        self.assertTrue(torch.allclose(imgs, orig_img.repeat_interleave(B, dim=0)))

        self.assertTrue(torch.allclose(img_ds.load_image_crop(), orig_img))

        # calling load image fails if the filepaths are not given
        with self.assertRaises(AttributeError) as e:
            _ = no_fps.load_image_crop()
        self.assertTrue(
            "Could not load image crops without proper filepaths given" in str(e.exception), msg=e.exception
        )

    def test_get_image_and_load(self):
        ds = DataSample(bbox=DUMMY_BBOX, filepath=DUMMY_FP)
        img = ds.image
        self.assertTrue(torch.allclose(img, load_test_image(IMG_NAME)))

    def test_get_image_crop_and_load(self):
        ds = DataSample(bbox=DUMMY_BBOX, crop_path=DUMMY_FP)
        crop = ds.image_crop
        self.assertTrue(torch.allclose(crop, load_test_image(IMG_NAME)))


class TestDataGetter(unittest.TestCase):
    def test_get_ds_data_getter(self):
        getter = get_ds_data_getter(["image", "filepath"])
        self.assertTrue(callable(getter))
        ds = DataSample(**DUMMY_DATA.copy())
        self.assertTrue(torch.allclose(getter(ds)[0], ds.image))


if __name__ == "__main__":
    unittest.main()
