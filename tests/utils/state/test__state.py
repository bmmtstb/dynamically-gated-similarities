import os
import unittest
from copy import deepcopy

import torch
from torchvision import tv_tensors

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.state import get_ds_data_getter, State
from dgs.utils.types import Device, FilePaths, Heatmap, Image
from tests.helper import load_test_image, test_multiple_devices

J = 20
J_DIM = 2
B = 10

PID = torch.tensor([13], dtype=torch.long)
PIDS = torch.ones(B, dtype=torch.long) * PID

IMG_NAME = "866-200x300.jpg"
DUMMY_IMG: Image = load_test_image(IMG_NAME)
DUMMY_IMG_BATCH: Image = tv_tensors.Image(torch.cat([DUMMY_IMG.detach().clone() for _ in range(B)]))

DUMMY_KP_TENSOR: torch.Tensor = torch.rand((1, J, J_DIM))
DUMMY_KP = DUMMY_KP_TENSOR.detach().clone()
DUMMY_KP_BATCH: torch.Tensor = torch.cat([DUMMY_KP_TENSOR.detach().clone() for _ in range(B)])

DUMMY_BBOX_TENSOR: torch.Tensor = torch.ones((1, 4))
DUMMY_BBOX: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    DUMMY_BBOX_TENSOR, format=tv_tensors.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000)
)
DUMMY_BBOX_BATCH: tv_tensors.BoundingBoxes = tv_tensors.BoundingBoxes(
    torch.cat([DUMMY_BBOX_TENSOR.detach().clone() for _ in range(B)]),
    format=tv_tensors.BoundingBoxFormat.XYWH,
    canvas_size=(1000, 1000),
)

DUMMY_FP_STRING: str = f"./tests/test_data/images/{IMG_NAME}"
DUMMY_FP: FilePaths = (os.path.normpath(os.path.join(PROJECT_ROOT, "./tests/test_data/images/" + IMG_NAME)),)
DUMMY_FP_BATCH: FilePaths = tuple(os.path.normpath(os.path.join(PROJECT_ROOT, DUMMY_FP_STRING)) for _ in range(B))

DUMMY_HM_TENSOR: torch.Tensor = torch.distributions.uniform.Uniform(0, 1).sample(torch.Size((1, J, B, 20))).float()
DUMMY_HM: Heatmap = tv_tensors.Mask(DUMMY_HM_TENSOR.detach().clone(), dtype=torch.float32)
DUMMY_HM_BATCH: Heatmap = tv_tensors.Mask(
    torch.cat([DUMMY_HM_TENSOR.detach().clone() for _ in range(B)]), dtype=torch.float32
)

DUMMY_WEIGHT: torch.Tensor = torch.tensor([i / J for i in range(J)]).view((1, J, 1)).float()
DUMMY_WEIGHT_BATCH: torch.Tensor = torch.cat([DUMMY_WEIGHT.detach().clone() for _ in range(B)]).float()

DUMMY_DATA: dict[str, any] = {
    "filepath": DUMMY_FP,
    "crop_path": DUMMY_FP,
    "bbox": DUMMY_BBOX,
    "keypoints": DUMMY_KP,
    "keypoints_local": DUMMY_KP,
    "heatmap": DUMMY_HM,
    "image": DUMMY_IMG,
    "image_crop": DUMMY_IMG,
    "person_id": PID,
    "class_id": PID,
    "track_id": PID,
    "joint_weight": DUMMY_WEIGHT,
}

DUMMY_DATA_BATCH: dict[str, any] = {
    "filepath": DUMMY_FP_BATCH,
    "crop_path": DUMMY_FP_BATCH,
    "bbox": DUMMY_BBOX_BATCH,
    "keypoints": DUMMY_KP_BATCH,
    "keypoints_local": DUMMY_KP_BATCH,
    "heatmap": DUMMY_HM_BATCH,
    "image": DUMMY_IMG_BATCH,
    "image_crop": DUMMY_IMG_BATCH,
    "person_id": PIDS,
    "class_id": PIDS,
    "track_id": PIDS,
    "joint_weight": DUMMY_WEIGHT_BATCH,
}


class TestState(unittest.TestCase):

    def test_init_regular(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):
                ds = State(bbox=DUMMY_BBOX, validate=validate)
                self.assertTrue(torch.allclose(ds.bbox, DUMMY_BBOX))
                self.assertEqual(len(ds), 1)
                self.assertEqual(ds.B, 1)

                multi_ds = State(bbox=DUMMY_BBOX_BATCH, validate=validate)
                self.assertTrue(torch.allclose(multi_ds.bbox, DUMMY_BBOX_BATCH))
                self.assertEqual(len(multi_ds), B)
                self.assertEqual(multi_ds.B, B)

    def test_init_with_unknown_kwarg(self):
        ds = State(bbox=DUMMY_BBOX, dummy="dummy")
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
                DUMMY_BBOX_BATCH,
                DUMMY_KP_BATCH,
                DUMMY_FP_BATCH,
                DUMMY_BBOX_BATCH,
                DUMMY_KP_BATCH,
                True,
            ),
            (DUMMY_FP_BATCH, DUMMY_BBOX_BATCH, DUMMY_KP_BATCH, DUMMY_FP_BATCH, DUMMY_BBOX_BATCH, DUMMY_KP_BATCH, True),
        ]:
            with self.subTest(msg=f"v: {validate}, len: {len(bbox)} fp: {fp}, bbox: {bbox}, kp: {kp}"):
                ds = State(filepath=fp, bbox=bbox, keypoints=kp, validate=validate)
                self.assertEqual(ds.filepath, out_fp)
                self.assertTrue(torch.allclose(ds.bbox, out_bbox))
                self.assertTrue(torch.allclose(ds.keypoints, out_kp))
                self.assertEqual(ds.J, J)
                self.assertEqual(ds.joint_dim, J_DIM)

    def test_args_raises(self):
        with self.assertRaises(NotImplementedError) as e:
            _ = State("dummy", bbox=DUMMY_BBOX)
        self.assertTrue("Unknown arguments" in str(e.exception), msg=e.exception)


class TestStateAttributes(unittest.TestCase):

    def test_J(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):

                faulty_ds = State(bbox=DUMMY_BBOX, validate=validate)
                with self.assertRaises(NotImplementedError) as e:
                    _ = faulty_ds.J
                self.assertTrue("no global or local key-points in this object" in str(e.exception), msg=e.exception)

            for scope in ["keypoints", "keypoints_local"]:
                with self.subTest(msg="scope: {}, validate: {}".format(scope, validate)):

                    ds = State(bbox=DUMMY_BBOX, keypoints=DUMMY_KP, validate=validate)
                    self.assertEqual(ds.J, J)

                    multi_ds = State(bbox=DUMMY_BBOX_BATCH, keypoints=DUMMY_KP_BATCH, validate=validate)
                    self.assertEqual(multi_ds.J, J)

    def test_J_value_later(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):

                s = State(bbox=DUMMY_BBOX, validate=validate)
                with self.assertRaises(NotImplementedError) as e:
                    _ = s.J
                self.assertTrue("no global or local key-points in this object" in str(e.exception), msg=e.exception)

                s.keypoints = DUMMY_KP
                self.assertEqual(s.J, J)

    def test_joint_dim(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):

                faulty_ds = State(bbox=DUMMY_BBOX, validate=validate)
                with self.assertRaises(NotImplementedError) as e:
                    _ = faulty_ds.joint_dim
                self.assertTrue("no global or local key-points in this object" in str(e.exception), msg=e.exception)

            for scope in ["keypoints", "keypoints_local"]:
                with self.subTest(msg="scope: {}, validate: {}".format(scope, validate)):

                    ds = State(**{"bbox": DUMMY_BBOX, scope: DUMMY_KP, "validate": validate})
                    self.assertEqual(ds.joint_dim, J_DIM)

                    multi_ds = State(**{"bbox": DUMMY_BBOX_BATCH, scope: DUMMY_KP_BATCH, "validate": validate})
                    self.assertEqual(multi_ds.joint_dim, J_DIM)

    def test_joint_dim_later(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):

                s = State(bbox=DUMMY_BBOX, validate=validate)
                with self.assertRaises(NotImplementedError) as e:
                    _ = s.joint_dim
                self.assertTrue("no global or local key-points in this object" in str(e.exception), msg=e.exception)

                s.keypoints = DUMMY_KP
                self.assertEqual(s.joint_dim, J_DIM)

    def test_keypoints(self):
        scopes = ["keypoints", "keypoints_local"]
        for validate in [True, False]:
            for i, scope in enumerate(scopes):
                with self.subTest(msg="scope: {}, validate: {}".format(scope, validate)):
                    ds = State(**{"bbox": DUMMY_BBOX, scope: DUMMY_KP, "validate": validate})
                    setattr(ds, scopes[(i + 1) % 2], DUMMY_KP)

    def test_setting_bbox_fails(self):
        ds = State(**DUMMY_DATA)
        with self.assertRaises(NotImplementedError) as e:
            ds.bbox = DUMMY_BBOX
        self.assertTrue("not allowed to change the bounding box of an already" in str(e.exception), msg=e.exception)

    def test_filepath_exceptions(self):
        # tuple
        with self.assertRaises(ValueError) as e:
            _ = State(bbox=DUMMY_BBOX, filepath=DUMMY_FP + DUMMY_FP)
        self.assertTrue(
            "filepath must have the same number of entries as bounding-boxes. Got 2, expected 1" in str(e.exception),
            msg=e.exception,
        )
        # string
        with self.assertRaises(ValueError) as e:
            _ = State(bbox=DUMMY_BBOX_BATCH, filepath=DUMMY_FP_STRING)
        self.assertTrue(f"Got a single path, expected {B}" in str(e.exception), msg=e.exception)

    def test_get_filepath_fails_as_string(self):
        ds = State(bbox=DUMMY_BBOX)
        ds.data["filepath"] = DUMMY_FP_STRING
        with self.assertRaises(AssertionError) as e:
            _ = ds.filepath
        self.assertTrue("filepath must be a tuple but got" in str(e.exception), msg=e.exception)

    def test_class_id(self):
        ds = State(bbox=DUMMY_BBOX, validate=False, class_id=1)
        self.assertEqual(ds.class_id.ndim, 1)
        self.assertEqual(ds.class_id, torch.ones(1, dtype=torch.long))

        ds = State(bbox=DUMMY_BBOX, validate=False, class_id=torch.ones(1))
        self.assertEqual(ds.class_id.ndim, 1)
        self.assertEqual(ds.class_id, torch.ones(1, dtype=torch.long))

    def test_person_id(self):
        ds = State(bbox=DUMMY_BBOX, validate=False, person_id=1)
        self.assertEqual(ds.person_id.ndim, 1)
        self.assertEqual(ds.person_id, torch.ones(1, dtype=torch.long))

        ds = State(bbox=DUMMY_BBOX, validate=False, person_id=torch.ones(1))
        self.assertEqual(ds.person_id, torch.ones(1, dtype=torch.long))
        self.assertEqual(ds.person_id, torch.ones(1, dtype=torch.long))

    def test_track_id(self):
        ds = State(bbox=DUMMY_BBOX, validate=False, track_id=1)
        self.assertEqual(ds.track_id.ndim, 1)
        self.assertEqual(ds.track_id, torch.ones(1, dtype=torch.long))

        ds = State(bbox=DUMMY_BBOX, validate=False, track_id=torch.ones(1))
        self.assertEqual(ds.track_id.ndim, 1)
        self.assertEqual(ds.track_id, torch.ones(1, dtype=torch.long))

    def test_crop_path(self):
        ds = State(bbox=DUMMY_BBOX, validate=False, crop_path=("dummy",))
        self.assertEqual(ds.crop_path, ("dummy",))

    @test_multiple_devices
    def test_init_with_device(self, device: Device):
        out_id = PID.detach().clone().to(device=device)
        out_image = DUMMY_IMG.detach().clone().to(device=device)
        out_imgcrop = DUMMY_IMG.detach().clone().to(device=device)
        out_kp = DUMMY_KP.detach().clone().to(device=device)
        out_loc_kp = DUMMY_KP.detach().clone().to(device=device)
        out_bbox = DUMMY_BBOX.detach().clone().to(device=device)
        out_joint_weight = DUMMY_WEIGHT.detach().clone().to(device=device)

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

                    ds: State = State(**data_dict)
                    self.assertEqual(ds.device, torch.device(device))
                    self.assertTrue(torch.allclose(ds.person_id, out_id))
                    self.assertTrue(torch.allclose(ds.image, out_image))
                    self.assertTrue(torch.allclose(ds.image_crop, out_imgcrop))
                    self.assertTrue(torch.allclose(ds.keypoints, out_kp))
                    self.assertTrue(torch.allclose(ds.keypoints_local, out_loc_kp))
                    self.assertTrue(torch.allclose(ds.bbox, out_bbox))
                    self.assertTrue(torch.allclose(ds.joint_weight, out_joint_weight))
                    self.assertEqual(ds.J, J)
                    self.assertEqual(ds.joint_dim, J_DIM)


class TestStateFunctions(unittest.TestCase):

    def test_len(self):
        for fps, length in [
            (os.path.join(PROJECT_ROOT, DUMMY_FP_STRING), 1),
            ((os.path.join(PROJECT_ROOT, DUMMY_FP_STRING),), 1),
        ]:
            with self.subTest(msg="fps: {}, length: {}".format(fps, length)):
                ds = State(bbox=DUMMY_BBOX, validate=False)
                self.assertEqual(len(ds), length)

        multi_ds = State(bbox=DUMMY_BBOX_BATCH)
        self.assertEqual(len(multi_ds), B)

    def test_equality(self):
        for d1, d2, eq in [
            ({"bbox": DUMMY_BBOX}, {"bbox": DUMMY_BBOX}, True),
            ({"bbox": DUMMY_BBOX, "validate": True}, {"bbox": DUMMY_BBOX, "validate": False}, False),
            (DUMMY_DATA, DUMMY_DATA, True),
        ]:
            with self.subTest(msg="d1: {}, d2: {}, eq: {}".format(d1, d2, eq)):
                s1 = State(**d1)
                s2 = State(**d2)
                self.assertEqual(s1 == s2, eq)
        self.assertFalse(State(**DUMMY_DATA) == "dummy")

    def test_copy(self):
        kp = DUMMY_KP
        s1 = State(bbox=DUMMY_BBOX, keypoints=kp)
        s2 = s1.copy()

        self.assertTrue(isinstance(s2, State))
        self.assertTrue(torch.allclose(s2.keypoints, s1.keypoints))

        s2.keypoints += torch.ones_like(DUMMY_KP)

        self.assertTrue(torch.allclose(s2.keypoints, DUMMY_KP + torch.ones_like(DUMMY_KP)))
        self.assertTrue(torch.allclose(s1.keypoints, DUMMY_KP))

    @test_multiple_devices
    def test_to(self, device: torch.device):
        bbox = tv_tensors.wrap(DUMMY_BBOX.cpu(), like=DUMMY_BBOX)
        kp = DUMMY_KP.cpu()
        cid = torch.ones(1, dtype=torch.long).cpu()
        ds = State(bbox=bbox, keypoints=kp, class_id=cid, validate=False)
        ds.to(device=device)
        self.assertEqual(ds.bbox.device, device)
        self.assertEqual(ds.keypoints.device, device)
        self.assertEqual(ds.class_id.device, device)

    @test_multiple_devices
    def test_extract_and_split(self, device: torch.device):
        for states, res_states in [
            (State(**DUMMY_DATA, device=device), [State(**DUMMY_DATA, device=device)]),
            (
                State(**DUMMY_DATA_BATCH, device=device),
                [State(**DUMMY_DATA, device=device) for _ in range(B)],
            ),
            (
                State(
                    bbox=tv_tensors.BoundingBoxes(
                        torch.stack([torch.tensor([i, i, 7, 9]) for i in range(B)]), canvas_size=(10, 10), format="XYWH"
                    ),
                    device=device,
                ),
                [
                    State(
                        bbox=tv_tensors.BoundingBoxes(torch.tensor([i, i, 7, 9]), canvas_size=(10, 10), format="XYWH"),
                        device=device,
                    )
                    for i in range(B)
                ],
            ),
            (
                State(bbox=DUMMY_BBOX_BATCH, filepath=tuple(DUMMY_FP_STRING for _ in range(B)), device=device),
                [State(bbox=DUMMY_BBOX, filepath=(DUMMY_FP_STRING,), device=device) for _ in range(B)],
            ),
            (
                State(bbox=DUMMY_BBOX_BATCH, dummy=[str(i) for i in range(B)], device=device),
                [State(bbox=DUMMY_BBOX, dummy=[str(i)], device=device) for i in range(B)],
            ),
            (
                State(bbox=DUMMY_BBOX_BATCH, tensor=torch.ones(B), device=device),
                [State(bbox=DUMMY_BBOX, tensor=torch.ones(1), device=device) for _ in range(B)],
            ),
            (
                State(bbox=DUMMY_BBOX_BATCH, val_tensor=torch.tensor(2), device=device),
                [State(bbox=DUMMY_BBOX, val_tensor=torch.tensor(2), device=device) for _ in range(B)],
            ),
            (
                State(bbox=DUMMY_BBOX_BATCH, dict={"a": 1}, device=device),
                [State(bbox=DUMMY_BBOX, dict={"a": 1}, device=device) for _ in range(B)],
            ),
            (
                State(bbox=DUMMY_BBOX_BATCH, str="dummy", device=device),
                [State(bbox=DUMMY_BBOX, str="dummy", device=device) for _ in range(B)],
            ),
        ]:
            keys = list(states.keys())

            split = states.split()
            self.assertEqual(split, res_states, "test split")

            B_ = len(states)
            for i in range(-B_, B_):
                with self.subTest(msg="i: {}, B_: {}, device: {}, states-keys: {}".format(i, B_, device, keys)):
                    res = states.extract(i)
                    s_i = res_states[i]
                    self.assertTrue(isinstance(res, State))
                    self.assertEqual(res, s_i, "extracted equals result")
                    self.assertEqual(res.device, device, "test extracted device")
                    self.assertEqual(s_i.device, device, "test result device")

    def test_split_resulting_sizes(self):
        s = State(**DUMMY_DATA_BATCH)
        res = s.split()
        for r in res:
            # check the number of dimensions
            self.assertEqual(r.image.ndim, 4)
            self.assertEqual(r.image_crop.ndim, 4)
            self.assertEqual(r.joint_weight.ndim, 3)
            self.assertEqual(r.keypoints.ndim, 3)
            self.assertEqual(r.keypoints_local.ndim, 3)
            self.assertEqual(r.person_id.ndim, 1)
            self.assertEqual(r.class_id.ndim, 1)
            self.assertEqual(r.track_id.ndim, 1)

            # check that the first dimension is B
            self.assertEqual(r.image_crop.size(0), 1)
            self.assertEqual(r.joint_weight.size(0), 1)
            self.assertEqual(r.keypoints.size(0), 1)
            self.assertEqual(r.keypoints_local.size(0), 1)
            self.assertEqual(r.person_id.size(0), 1)
            self.assertEqual(r.class_id.size(0), 1)
            self.assertEqual(r.track_id.size(0), 1)

            # check non torch objects
            self.assertEqual(len(r.filepath), 1)
            self.assertEqual(len(r.crop_path), 1)

    def test_extract_errors(self):
        s = State(**DUMMY_DATA)
        with self.assertRaises(IndexError) as e:
            _ = s.extract(1)
        self.assertTrue("Expected index to lie within (-1, 0), but got: 1" in str(e.exception), msg=e.exception)

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
                    ds = State(**data)
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
        ds = State(filepath=(DUMMY_FP,), bbox=DUMMY_BBOX, validate=False)
        multi_ds = State(filepath=DUMMY_FP_BATCH, bbox=DUMMY_BBOX_BATCH, validate=False)
        no_fps = State(bbox=DUMMY_BBOX, validate=False)
        empty_fps = State(
            bbox=tv_tensors.BoundingBoxes(torch.empty((0, 4)), canvas_size=(0, 0), format="XYXY"),
            filepath=tuple(),
            validate=False,
        )
        img_ds = State(bbox=DUMMY_BBOX, image=orig_img.clone(), validate=False)

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

        # call load image with zero-length image data
        empty_img = empty_fps.load_image()
        self.assertTrue(torch.allclose(empty_img, tv_tensors.Image(torch.empty((0, 0, 1, 1)))))

    def test_load_image_crop(self):
        orig_img = load_test_image(IMG_NAME)
        ds = State(bbox=DUMMY_BBOX, crop_path=(DUMMY_FP,), validate=False)
        multi_ds = State(bbox=DUMMY_BBOX_BATCH, crop_path=DUMMY_FP_BATCH, validate=False)
        no_fps = State(bbox=DUMMY_BBOX, validate=False)
        empty_fps = State(
            bbox=tv_tensors.BoundingBoxes(torch.empty((0, 4)), canvas_size=(0, 0), format="XYXY"),
            crop_path=tuple(),
            validate=False,
        )
        img_ds = State(bbox=DUMMY_BBOX, image_crop=orig_img.clone(), validate=False)

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

        # call load image with zero-length image data
        empty_crop = empty_fps.load_image_crop()
        self.assertTrue(torch.allclose(empty_crop, tv_tensors.Image(torch.empty((0, 0, 1, 1)))))

    def test_get_image_and_load(self):
        ds = State(bbox=DUMMY_BBOX, filepath=DUMMY_FP)
        img = ds.image
        self.assertTrue(torch.allclose(img, load_test_image(IMG_NAME)))

    def test_get_image_crop_and_load(self):
        ds = State(bbox=DUMMY_BBOX, crop_path=DUMMY_FP)
        crop = ds.image_crop
        self.assertTrue(torch.allclose(crop, load_test_image(IMG_NAME)))


class TestDataGetter(unittest.TestCase):
    def test_get_ds_data_getter(self):
        getter = get_ds_data_getter(["image", "filepath"])
        self.assertTrue(callable(getter))
        ds = State(**DUMMY_DATA.copy())
        self.assertTrue(torch.allclose(getter(ds)[0], ds.image))


if __name__ == "__main__":
    unittest.main()
