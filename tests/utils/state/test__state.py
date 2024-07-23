import os
import unittest
from copy import deepcopy

import numpy as np
import torch
import torch as t
from torchvision import tv_tensors as tvte

from dgs.utils.config import DEF_VAL
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.state import collate_states, get_ds_data_getter, State
from dgs.utils.types import Device
from tests.helper import load_test_image, test_multiple_devices
from . import *


class TestState(unittest.TestCase):

    def test_init_regular(self):
        for validate in [True, False]:
            with self.subTest(msg="validate: {}".format(validate)):
                ds = State(bbox=DUMMY_BBOX, validate=validate)
                self.assertTrue(t.allclose(ds.bbox, DUMMY_BBOX))
                self.assertEqual(len(ds), 1)
                self.assertEqual(ds.B, 1)

                multi_ds = State(bbox=DUMMY_BBOX_BATCH, validate=validate)
                self.assertTrue(t.allclose(multi_ds.bbox, DUMMY_BBOX_BATCH))
                self.assertEqual(len(multi_ds), B)
                self.assertEqual(multi_ds.B, B)

    def test_init_with_unknown_kwarg(self):
        ds = State(bbox=DUMMY_BBOX, dummy="dummy")
        self.assertEqual(ds["dummy"], "dummy")

    def test_init_with_none_value(self):
        ds = State(bbox=DUMMY_BBOX, value=None)
        self.assertFalse("value" in ds)

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
                self.assertTrue(t.allclose(ds.bbox, out_bbox))
                self.assertTrue(t.allclose(ds.keypoints, out_kp))
                self.assertEqual(ds.J, J)
                self.assertEqual(ds.joint_dim, J_DIM)

    def test_args_raises(self):
        with self.assertRaises(NotImplementedError) as e:
            _ = State("dummy", bbox=DUMMY_BBOX)
        self.assertTrue("Unknown arguments" in str(e.exception), msg=e.exception)

    def test_getitems(self):
        ds = State(bbox=DUMMY_BBOX)
        self.assertTrue(torch.allclose(ds["bbox"], DUMMY_BBOX))
        self.assertEqual(ds[0], ds[-1])

        states = State(bbox=DUMMY_BBOX_BATCH)
        self.assertEqual(states[0], states[-1])
        self.assertEqual(states[0], ds)
        self.assertEqual(states[0:3], collate_states([ds for _ in range(2)]))

        with self.assertRaises(NotImplementedError) as e:
            _ = ds[1.4]
        self.assertTrue("Expected item to be str, int or slice, got" in str(e.exception), msg=e.exception)


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

    @test_multiple_devices
    def test_get_keypoints_load_from_file(self, device):
        res_kp, res_weights = t.load(DUMMY_KP_PATH).to(device=device).reshape((1, J, J_DIM + 1)).split([2, 1], dim=-1)
        for validate in [True, False]:
            for scope_kp, scope_path, data_path in [
                ("keypoints", "keypoints_path", DUMMY_KP_PATH),
                ("keypoints_local", "keypoints_local_path", DUMMY_KP_PATH),
            ]:
                with self.subTest(
                    msg="s_kp: {}, s_p: {}, data_path: {}, val: {}".format(scope_kp, scope_path, data_path, validate)
                ):
                    ds = State(
                        **{
                            "bbox": DUMMY_BBOX.to(device=device),
                            scope_path: data_path,
                            "validate": validate,
                            "device": device,
                        }
                    )
                    self.assertEqual(ds[scope_path], data_path)
                    kp = getattr(ds, scope_kp)
                    self.assertEqual(kp.device, res_kp.device)
                    self.assertTrue(t.allclose(kp, res_kp))
                    self.assertEqual(kp.size(-1), 2)
                    self.assertTrue(t.allclose(ds.joint_weight, res_weights))

    @test_multiple_devices
    def test_get_keypoints_from_crop_path(self, device):
        res_kp, res_weights = t.load(DUMMY_KP_PATH).to(device=device).reshape((1, J, J_DIM + 1)).split([2, 1], dim=-1)
        for validate in [True, False]:
            for scope_kp, crop_path, data_path in [
                ("keypoints", "keypoints_path", DUMMY_KP_PATH_GLOB),
                ("keypoints_local", "keypoints_local_path", DUMMY_KP_PATH_GLOB),
            ]:
                with self.subTest(
                    msg="s_kp: {}, c_p: {}, data_path: {}, val: {}".format(scope_kp, crop_path, data_path, validate)
                ):
                    ds = State(
                        **{
                            "bbox": DUMMY_BBOX.to(device=device),
                            crop_path: data_path,
                            "validate": validate,
                            "device": device,
                        }
                    )
                    self.assertEqual(ds[crop_path], data_path)
                    kp = getattr(ds, scope_kp)
                    self.assertEqual(kp.device, res_kp.device)
                    self.assertTrue(t.allclose(kp, res_kp))
                    self.assertEqual(kp.size(-1), 2)
                    self.assertTrue(t.allclose(ds.joint_weight, res_weights))

    def test_get_keypoints_exceptions(self):
        ds = State(**{"bbox": DUMMY_BBOX})
        with self.assertRaises(KeyError) as e:
            _ = getattr(ds, "keypoints")
        self.assertTrue("There are no key-points in this object" in str(e.exception), msg=e.exception)
        with self.assertRaises(KeyError) as e:
            _ = getattr(ds, "keypoints_local")
        self.assertTrue("There are no local key-points in this object" in str(e.exception), msg=e.exception)

    @test_multiple_devices
    def test_setting_bbox(self, device: Device):
        orig_devices: list[Device] = [t.device("cpu")]
        if t.cuda.is_available():
            orig_devices.append(t.device("cuda:0"))

        for original_device in orig_devices:
            with self.subTest(msg=f"original: {original_device}, new_device: {device}"):
                ds = State(**DUMMY_DATA).to(device=original_device)
                self.assertEqual(ds.device, t.device(original_device))
                new_bbox = tvte.BoundingBoxes(
                    DUMMY_BBOX_TENSOR + 1, format=tvte.BoundingBoxFormat.XYWH, canvas_size=(1000, 1000), device=device
                )
                ds.bbox = new_bbox
                self.assertTrue(t.allclose(ds.bbox, new_bbox))
                self.assertEqual(t.device(ds.bbox.device), t.device(device))

    def test_setting_bbox_exceptions(self):
        ds = State(**DUMMY_DATA)
        with self.assertRaises(TypeError) as e:
            ds.bbox = DUMMY_BBOX_TENSOR
        self.assertTrue("Expected bounding box, got" in str(e.exception), msg=e.exception)
        with self.assertRaises(ValueError) as e:
            ds.bbox = DUMMY_BBOX_BATCH
        self.assertTrue("Can't switch bbox shape. Expected" in str(e.exception), msg=e.exception)

    def test_filepath(self):
        for validate in [True, False]:
            for fp, box, fp_res in [
                (DUMMY_FP, DUMMY_BBOX, DUMMY_FP),
                (DUMMY_FP_STRING, DUMMY_BBOX, DUMMY_FP),
                (DUMMY_FP_BATCH, DUMMY_BBOX_BATCH, DUMMY_FP_BATCH),
            ]:
                with self.subTest(msg="fp: {}, fp_res: {}".format(fp, fp_res)):
                    s = State(bbox=box, filepath=fp, validate=validate)
                    self.assertEqual(s.filepath, fp_res)

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
        ds["filepath"] = DUMMY_FP_STRING
        with self.assertRaises(AssertionError) as e:
            _ = ds.filepath
        self.assertTrue("filepath must be a tuple but got" in str(e.exception), msg=e.exception)

    def test_class_id(self):
        ds = State(bbox=DUMMY_BBOX, validate=False, class_id=1)
        self.assertEqual(ds.class_id.ndim, 1)
        self.assertEqual(ds.class_id, t.ones(1, dtype=t.long))

        ds = State(bbox=DUMMY_BBOX, validate=False, class_id=t.ones(1))
        self.assertEqual(ds.class_id.ndim, 1)
        self.assertEqual(ds.class_id, t.ones(1, dtype=t.long))

    def test_person_id(self):
        ds = State(bbox=DUMMY_BBOX, validate=False, person_id=1)
        self.assertEqual(ds.person_id.ndim, 1)
        self.assertEqual(ds.person_id, t.ones(1, dtype=t.long))

        ds = State(bbox=DUMMY_BBOX, validate=False, person_id=t.ones(1))
        self.assertEqual(ds.person_id, t.ones(1, dtype=t.long))
        self.assertEqual(ds.person_id, t.ones(1, dtype=t.long))

    def test_track_id(self):
        ds = State(bbox=DUMMY_BBOX, validate=False, track_id=1)
        self.assertEqual(ds.track_id.ndim, 1)
        self.assertEqual(ds.track_id, t.ones(1, dtype=t.long))

        ds = State(bbox=DUMMY_BBOX, validate=False, track_id=t.ones(1))
        self.assertEqual(ds.track_id.ndim, 1)
        self.assertEqual(ds.track_id, t.ones(1, dtype=t.long))

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
                    self.assertEqual(ds.device, t.device(device))
                    self.assertTrue(t.allclose(ds.person_id, out_id))
                    self.assertTrue(t.allclose(ds.image[0], out_image))
                    self.assertTrue(t.allclose(ds.image_crop[0], out_imgcrop))
                    self.assertTrue(t.allclose(ds.keypoints, out_kp))
                    self.assertTrue(t.allclose(ds.keypoints_local, out_loc_kp))
                    self.assertTrue(t.allclose(ds.bbox, out_bbox))
                    self.assertTrue(t.allclose(ds.joint_weight, out_joint_weight))
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
        self.assertTrue(t.allclose(s2.keypoints, s1.keypoints))

        s2.keypoints += t.ones_like(DUMMY_KP)

        self.assertTrue(t.allclose(s2.keypoints, DUMMY_KP + t.ones_like(DUMMY_KP)))
        self.assertTrue(t.allclose(s1.keypoints, DUMMY_KP))

    @test_multiple_devices
    def test_to(self, device: Device):
        bbox = tvte.wrap(DUMMY_BBOX.cpu(), like=DUMMY_BBOX)
        kp = DUMMY_KP.cpu()
        cid = t.ones(1, dtype=t.long).cpu()
        ds = State(bbox=bbox, keypoints=kp, class_id=cid, validate=False)
        ds.to(device=device)
        self.assertEqual(ds.bbox.device, device)
        self.assertEqual(ds.keypoints.device, device)
        self.assertEqual(ds.class_id.device, device)

    @test_multiple_devices
    def test_extract_and_split(self, device: Device):
        for validate in [True, False]:
            for states, res_states in [
                (
                    State(**DUMMY_DATA, device=device, validate=validate),
                    [State(**DUMMY_DATA, device=device, validate=validate)],
                ),
                (
                    State(**DUMMY_DATA_BATCH, device=device, validate=validate),
                    [State(**DUMMY_DATA, device=device, validate=validate) for _ in range(B)],
                ),
                (
                    State(
                        bbox=tvte.BoundingBoxes(
                            t.stack([t.tensor([i, i, 7, 9]) for i in range(B)]),
                            canvas_size=(10, 10),
                            format="XYWH",
                        ),
                        keypoints=DUMMY_KP_BATCH,
                        filepath=tuple(DUMMY_FP_STRING for _ in range(B)),
                        tensor=t.ones(B),
                        val_tensor=t.tensor(2, device=device),
                        tuple=tuple(str(i) for i in range(B)),
                        list=[str(i) for i in range(B)],
                        dict={"a": 1},
                        set={1, 2, 3},
                        str="dummy",
                        int=1,
                        numpy=np.ones(3),
                        device=device,
                        validate=validate,
                    ),
                    [
                        State(
                            bbox=tvte.BoundingBoxes(t.tensor([i, i, 7, 9]), canvas_size=(10, 10), format="XYWH"),
                            keypoints=DUMMY_KP,
                            filepath=(DUMMY_FP_STRING,),
                            tensor=t.ones(1),
                            val_tensor=t.tensor(2, device=device),
                            tuple=(str(i),),
                            list=[str(i)],
                            dict={"a": 1},
                            set={1, 2, 3},
                            str="dummy",
                            int=1,
                            numpy=np.asarray(1),
                            device=device,
                            validate=validate,
                        )
                        for i in range(B)
                    ],
                ),
            ]:
                keys = list(states.keys())

                with self.subTest(msg="device: {}, val: {}, states-keys: {}".format(device, validate, keys)):
                    split = states.split()
                    self.assertEqual(split, res_states, "test split")

                    B_ = len(states)
                    for i in range(-B_, B_):
                        res = states.extract(i)
                        s_i = res_states[i]
                        self.assertTrue(isinstance(res, State))
                        self.assertEqual(res, s_i, "extracted equals result")
                        self.assertEqual(res.device, device, "test extracted device")
                        self.assertEqual(s_i.device, device, "test result device")
                        self.assertEqual(res.keypoints.ndim, 3)
                        self.assertEqual(s_i.keypoints.ndim, 3)

    def test_split_resulting_sizes(self):
        s = State(**DUMMY_DATA_BATCH)
        res = s.split()
        for r in res:
            # check the number of dimensions
            self.assertTrue(all(img.ndim == 4 for img in r.image))
            self.assertEqual(r.image_crop.ndim, 4)
            self.assertEqual(r.joint_weight.ndim, 3)
            self.assertEqual(r.keypoints.ndim, 3)
            self.assertEqual(r.keypoints_local.ndim, 3)
            self.assertEqual(r.person_id.ndim, 1)
            self.assertEqual(r.class_id.ndim, 1)
            self.assertEqual(r.track_id.ndim, 1)

            # check that the first dimension is B
            self.assertEqual(len(r.image), 1)
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
                t.int32,
                t.round(t.tensor([i / J for i in range(J)]).view((1, J, 1)), decimals=0).to(dtype=t.int32),
            ),
            (
                DUMMY_WEIGHT.detach().clone(),
                1,
                t.float32,
                t.round(t.tensor([i / J for i in range(J)]).view((1, J, 1)), decimals=1),
            ),
            (  # typecast only
                DUMMY_WEIGHT.detach().clone(),
                -1,
                t.int32,
                t.tensor([i / J for i in range(J)]).view((1, J, 1)).to(dtype=t.int32),
            ),
            (  # start with int
                DUMMY_WEIGHT.detach().clone().int(),
                0,
                t.int32,
                t.round(t.tensor([i / J for i in range(J)], dtype=t.int32).view((1, J, 1)).float(), decimals=0).to(
                    dtype=t.int32
                ),
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
                    self.assertTrue(t.allclose(r, result))
                    if overwrite:
                        self.assertTrue(t.allclose(r, dsi.joint_weight))
                    else:
                        self.assertTrue(t.allclose(dsi.joint_weight, ds.joint_weight))

    def test_load_image(self):
        orig_img = load_test_image(IMG_NAME)
        s = State(filepath=(DUMMY_FP,), bbox=DUMMY_BBOX, validate=False)
        multi_s = State(filepath=DUMMY_FP_BATCH, bbox=DUMMY_BBOX_BATCH, validate=False)
        no_fps = State(bbox=DUMMY_BBOX, validate=False)
        empty_fps = State(
            bbox=tvte.BoundingBoxes(t.empty((0, 4)), canvas_size=(0, 0), format="XYXY"),
            filepath=tuple(),
            validate=False,
        )
        img_ds = State(bbox=DUMMY_BBOX, image=orig_img.clone(), validate=False)

        # get using data -> fails if not present yet
        for obj in [s, multi_s, no_fps]:
            with self.subTest(msg="obj: {}".format(obj)):
                with self.assertRaises(KeyError) as e:
                    _ = obj.copy()["image"]
                self.assertTrue("'image'" in str(e.exception), msg=e.exception)
        # -> succeed if present
        img_0 = img_ds["image"][0]
        self.assertTrue(t.allclose(img_0, orig_img))

        # call load_image
        s.load_image(store=False)
        self.assertTrue("image" not in s)
        s.load_image(store=True)
        self.assertTrue("image" in s)
        imgs_1 = s["image"]
        self.assertTrue(isinstance(imgs_1, list))
        img_1 = imgs_1[0]
        self.assertTrue(isinstance(img_1, tvte.Image))
        self.assertEqual(img_1.shape, orig_img.shape)
        self.assertTrue(t.allclose(img_1, orig_img))

        imgs_2 = multi_s.load_image()
        self.assertTrue(isinstance(imgs_2, list))
        for img_2 in imgs_2:
            self.assertTrue(isinstance(img_2, tvte.Image))
            self.assertEqual(list(img_2.shape), [1] + list(orig_img.shape)[-3:])
            self.assertTrue(t.allclose(img_2, orig_img))

        self.assertTrue(t.allclose(img_ds.load_image()[0], orig_img))

        # calling load image fails if the filepaths are not given
        with self.assertRaises(AttributeError) as e:
            _ = no_fps.load_image()
        self.assertTrue("Could not load images without proper filepaths given" in str(e.exception), msg=e.exception)

        # call load image with zero-length image data
        empty_img = empty_fps.load_image()
        self.assertEqual(empty_img, [])

    def test_get_image_and_load(self):
        s = State(bbox=DUMMY_BBOX, filepath=DUMMY_FP)
        imgs = s.image
        self.assertTrue("image" not in s)
        self.assertTrue(all(t.allclose(i, load_test_image(IMG_NAME)) for i in imgs))

    def test_get_image_crop_and_load(self):
        s = State(bbox=DUMMY_BBOX, crop_path=DUMMY_FP)
        crops = s.image_crop
        self.assertTrue("image_crop" not in s)
        self.assertTrue(all(t.allclose(i, load_test_image(IMG_NAME)) for i in crops))

    def test_clean(self):
        s = State(**DUMMY_DATA)
        multi_s = State(**DUMMY_DATA_BATCH)

        self.assertTrue("image" in s)
        self.assertTrue("image_crop" in s)
        s.clean()
        self.assertTrue("image" not in s)
        self.assertTrue("image_crop" not in s)

        self.assertTrue("image" in multi_s)
        self.assertTrue("image_crop" in multi_s)
        multi_s.clean(["image", "image_crop", "embedding", "keypoints"])
        self.assertTrue("image" not in multi_s)
        self.assertTrue("image_crop" not in multi_s)
        self.assertTrue("embedding" not in multi_s)
        self.assertTrue("keypoints" not in multi_s)

        with self.assertRaises(ValueError) as e:
            _ = s.clean("bbox")
        self.assertTrue("Cannot clean bounding box!" in str(e.exception), msg=e.exception)


class TestImageCrop(unittest.TestCase):
    orig_img = load_test_image(IMG_NAME)

    def test_load_image_crop(self):
        s = State(bbox=DUMMY_BBOX, crop_path=DUMMY_FP, validate=False)

        # get using data -> fails if not present yet
        with self.assertRaises(KeyError) as e:
            _ = s.copy()["image_crop"]
        self.assertTrue("'image_crop'" in str(e.exception), msg=e.exception)

        # s - call load_image_crop
        s.load_image_crop(store=False)
        self.assertTrue("image_crop" not in s)
        self.assertTrue("keypoints_local" not in s)
        s.load_image_crop(store=True)
        self.assertTrue("image_crop" in s)
        self.assertTrue("keypoints_local" in s)
        crop = s["image_crop"]
        self.assertTrue(isinstance(crop, tvte.Image))
        self.assertEqual(crop.shape, self.orig_img.shape)
        self.assertTrue(t.allclose(crop, self.orig_img))

    def test_load_image_crop_with_given_crop(self):
        # -> succeed if present
        img_ds = State(bbox=DUMMY_BBOX, image_crop=self.orig_img.clone(), keypoints_local=DUMMY_KP, validate=False)
        self.assertTrue(t.allclose(img_ds["image_crop"], self.orig_img))
        self.assertTrue(t.allclose(img_ds.load_image_crop()[0], self.orig_img))

    def test_load_image_crop_batched(self):
        multi_s = State(bbox=DUMMY_BBOX_BATCH, crop_path=DUMMY_FP_BATCH, validate=False)
        imgs = multi_s.load_image_crop()
        self.assertTrue(isinstance(imgs, tvte.Image))
        self.assertEqual(list(imgs.shape), [B] + list(self.orig_img.shape)[1:])
        self.assertTrue(t.allclose(imgs, self.orig_img.repeat_interleave(B, dim=0)))

    def test_load_image_crop_fails_without_path(self):
        # calling load image fails if the filepaths aren't given
        no_fps = State(bbox=DUMMY_BBOX, validate=False)
        with self.assertRaises(AttributeError) as e:
            _ = no_fps.load_image_crop()
        self.assertTrue(
            "Could not load image crops without either a proper filepath given or an image and bbox given."
            in str(e.exception),
            msg=e.exception,
        )

    def test_load_image_crop_with_zero_length(self):
        # call load image with zero-length image data
        empty_fps = State(
            bbox=tvte.BoundingBoxes(t.empty((0, 4)), canvas_size=(0, 0), format="XYXY"),
            crop_path=tuple(),
            validate=False,
        )
        empty_crop = empty_fps.load_image_crop()
        self.assertEqual(empty_crop, [])

    def test_load_image_crop_without_associated_kp_file(self):
        # make sure local kp don't get set if the crop doesn't have an associated kp file
        s = State(
            bbox=DUMMY_BBOX,
            crop_path=(
                os.path.join(PROJECT_ROOT, "./tests/test_data/images/866-200x300.jpg"),
            ),  # there is no respective .pt file
            keypoints=DUMMY_KP,
            validate=False,
        )
        s.load_image_crop(store=False)
        self.assertTrue("image_crop" not in s)
        self.assertTrue("keypoints_local" not in s)
        s.load_image_crop(store=True)
        self.assertTrue("image_crop" in s)
        self.assertTrue("keypoints_local" in s)

    def test_load_img_crop_by_extraction(self):
        s = State(bbox=DUMMY_BBOX, filepath=DUMMY_FP, keypoints=DUMMY_KP)

        crop = s.load_image_crop(store=False)
        self.assertTrue(isinstance(crop, tvte.Image))
        self.assertEqual(crop.shape, t.Size((1, 3, *DEF_VAL["images"]["crop_size"])))
        self.assertTrue("keypoints_local" not in s)
        self.assertTrue("image_crop" not in s)

        crop = s.load_image_crop(store=True)
        self.assertTrue("keypoints_local" in s)
        self.assertTrue("image_crop" in s)
        self.assertTrue(isinstance(crop, tvte.Image))
        self.assertEqual(crop.shape, t.Size((1, 3, *DEF_VAL["images"]["crop_size"])))
        self.assertEqual(s.keypoints_local.shape, DUMMY_KP.shape)

    def test_load_img_crop_by_extraction_crop_size(self):
        s = State(bbox=DUMMY_BBOX, filepath=DUMMY_FP, keypoints=DUMMY_KP)
        out_size = (100, 100)

        crop = s.load_image_crop(store=False, crop_size=out_size)
        self.assertTrue(isinstance(crop, tvte.Image))
        self.assertEqual(crop.shape, t.Size((1, 3, *out_size)))
        self.assertTrue("keypoints_local" not in s)
        self.assertTrue("image_crop" not in s)

        crop = s.load_image_crop(store=True, crop_size=out_size)
        self.assertTrue("keypoints_local" in s)
        self.assertTrue("image_crop" in s)
        self.assertTrue(isinstance(crop, tvte.Image))
        self.assertEqual(crop.shape, t.Size((1, 3, *out_size)))
        self.assertEqual(s.keypoints_local.shape, DUMMY_KP.shape)

    def test_load_img_crop_by_extraction_batched(self):
        s = State(bbox=DUMMY_BBOX_BATCH, filepath=DUMMY_FP_BATCH)

        crops = s.load_image_crop(store=False)
        self.assertTrue(isinstance(crops, tvte.Image))
        self.assertEqual(crops.shape, t.Size((B, 3, *DEF_VAL["images"]["crop_size"])))
        self.assertFalse("keypoints_local" in s)

        crops = s.load_image_crop(store=True)
        self.assertTrue(isinstance(crops, tvte.Image))
        self.assertEqual(crops.shape, t.Size((B, 3, *DEF_VAL["images"]["crop_size"])))
        self.assertFalse("keypoints_local" in s)  # no key points given

    def test_load_img_crop_by_extraction_batched_with_crop_size(self):
        s = State(bbox=DUMMY_BBOX_BATCH, filepath=DUMMY_FP_BATCH)
        out_size = (100, 100)

        crops = s.load_image_crop(store=False, crop_size=out_size)
        self.assertTrue(isinstance(crops, tvte.Image))
        self.assertEqual(crops.shape, t.Size((B, 3, *out_size)))
        self.assertFalse("keypoints_local" in s)

        crops = s.load_image_crop(store=True, crop_size=out_size)
        self.assertTrue(isinstance(crops, tvte.Image))
        self.assertEqual(crops.shape, t.Size((B, 3, *out_size)))
        self.assertFalse("keypoints_local" in s)  # no key points given


class TestDataGetter(unittest.TestCase):
    def test_get_ds_data_getter(self):
        getter = get_ds_data_getter(["bbox", "filepath"])
        self.assertTrue(callable(getter))
        s = State(**DUMMY_DATA.copy())
        self.assertTrue(t.allclose(getter(s)[0], s.bbox))
        self.assertEqual(getter(s)[1], s.filepath)


if __name__ == "__main__":
    unittest.main()
