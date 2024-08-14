import os.path
import shutil
import unittest

from dgs.models.dataset.posetrack21 import (
    get_pose_track_21,
    PoseTrack21_BBox,
    PoseTrack21_Image,
    validate_pt21_json,
)
from dgs.models.loader import get_data_loader
from dgs.utils.config import load_config
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import is_abs_dir, mkdir_if_missing
from dgs.utils.state import EMPTY_STATE, State
from dgs.utils.utils import HidePrint


class TestPT21Helpers(unittest.TestCase):
    def test_validate_pt21_json(self):
        with self.assertRaises(ValueError) as e:
            # noinspection PyTypeChecker
            validate_pt21_json("")
        self.assertTrue("The PoseTrack21 json file is expected to be a dict" in str(e.exception), msg=e.exception)

        with self.assertRaises(KeyError) as e:
            validate_pt21_json({"annotations": []})
        self.assertTrue("PoseTrack21 .json file has an images key" in str(e.exception), msg=e.exception)

        with self.assertRaises(KeyError) as e:
            validate_pt21_json({"images": []})
        self.assertTrue("PoseTrack21 .json file has an annotations key" in str(e.exception), msg=e.exception)

    # def test_subm_data_from_state(self):
    #     s = State(**DUMMY_DATA, validate=False)
    #     img, anno = submission_data_from_state(s)
    #     self.assertTrue(isinstance(img, dict))
    #     self.assertTrue(isinstance(anno, list))
    #     self.assertEqual(len(anno), 1)
    #
    #     states = State(**DUMMY_DATA_BATCH, validate=False)
    #     img_, annos = submission_data_from_state(states)
    #     self.assertTrue(isinstance(img, dict))
    #     self.assertTrue(isinstance(anno, list))
    #     self.assertEqual(len(annos), B)
    #
    #     self.assertEqual(img, img_)
    #     self.assertTrue(all(a == anno[0] for a in annos))
    #     self.assertEqual(len(annos[0]["kps"]), 17 * 3)
    #
    #     # early return with B == 0
    #     empty_state = State(
    #         bbox=tvte.BoundingBoxes(torch.empty((0, 4)), canvas_size=(0, 0), format="XYXY"),
    #         validate=False,
    #         filepath=DUMMY_FP_STRING,
    #         image_id=torch.ones((1, 1)),
    #         frame_id=torch.ones((1, 1)),
    #     )
    #     empty_img, empty_anno = submission_data_from_state(empty_state)
    #     self.assertTrue(isinstance(img, dict))
    #     self.assertTrue(isinstance(anno, list))
    #     self.assertEqual(len(empty_anno), 0)
    #     self.assertEqual(empty_img["file_name"], DUMMY_FP_STRING)
    #     self.assertEqual(empty_img["id"], 1)
    #     self.assertEqual(empty_img["frame_id"], 1)
    #
    # def test_subm_data_from_s_except(self):
    #     VALID_IMG_DATA = {"bbox": DUMMY_BBOX, "filepath": DUMMY_FP, "image_id": [1], "frame_id": [1]}
    #     for data, exp_type, msg in [
    #         ({"bbox": DUMMY_BBOX}, KeyError, "Expected key 'filepath' to be in State."),
    #         (
    #             {"bbox": DUMMY_BBOX, "filepath": DUMMY_FP, "image_id": [1, 2]},
    #             ValueError,
    #             "Expected 'image_id' (2) to have a length of exactly 1.",
    #         ),
    #         (
    #             {"bbox": DUMMY_BBOX_BATCH, "filepath": DUMMY_FP_BATCH, "image_id": [1, 2]},
    #             ValueError,
    #             "State has different image_ids, expected all image_ids to match. got: '[1, 2]'.",
    #         ),
    #         (
    #             {"bbox": DUMMY_BBOX_BATCH, "filepath": DUMMY_FP_BATCH, "image_id": [1, 1, 1]},
    #             ValueError,
    #             f"Expected 'image_id' (3) to have the same length as the State ({B}).",
    #         ),
    #         # anno validation
    #         (VALID_IMG_DATA, KeyError, f"Expected key 'person_id' to be in State."),
    #         (
    #             {**VALID_IMG_DATA, "person_id": torch.tensor([1, 2])},
    #             ValueError,
    #             f"Expected 'person_id' (2) to have the same length as the State (1).",
    #         ),
    #     ]:
    #         with self.subTest(msg="msg: {}, exp_type: {}, data: {}".format(msg, exp_type, data)):
    #             s = State(**data, validate=False)
    #             with self.assertRaises(exp_type) as e:
    #                 _ = submission_data_from_state(s)
    #             self.assertTrue(msg in str(e.exception), msg=e.exception)
    #


class TestPoseTrack21BBoxDataset(unittest.TestCase):
    def test_init_directly(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        with HidePrint():
            ds = PoseTrack21_BBox(config=cfg, path=["test_single_dataset_1"])
        self.assertEqual(len(ds), 1)

    def test_init_single(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        with HidePrint():
            ds = get_pose_track_21(config=cfg, path=["test_single_dataset_1"])
        self.assertEqual(len(ds), 1)

    def test_init_multi(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        with HidePrint():
            ds = get_pose_track_21(config=cfg, path=["test_multi_dataset"])
        self.assertEqual(len(ds), 5 + 5 + 1)

    def test_init_folder(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        with HidePrint():
            ds = get_pose_track_21(config=cfg, path=["test_directory_dataset"])
        self.assertEqual(len(ds), 5 + 1)

    def test_init_folder_reshape_exception(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        with self.assertRaises(ValueError) as e:
            with HidePrint():
                _ = get_pose_track_21(config=cfg, path=["test_json_dataset_multi_images_without_reshape"])
        self.assertTrue(
            "The images within a single dataset should have equal shapes" in str(e.exception), msg=e.exception
        )

    def test_get_item(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        with HidePrint():
            ds = PoseTrack21_BBox(config=cfg, path=["test_single_dataset_1"])
        r = ds[0]
        self.assertTrue(isinstance(r, State))
        self.assertEqual(len(r), 1)

    def test_dataloader(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        B = int(cfg["test_dataloader_bbox"]["batch_size"])
        with HidePrint():
            dl = get_data_loader(config=cfg, path=["test_dataloader_bbox"])

        batch: State
        for i, batch in enumerate(dl):
            with self.subTest(msg=f"i: {i}, batch: {batch}"):

                self.assertTrue(isinstance(batch, State))
                self.assertEqual(len(batch), B)

                # check the number of dimensions
                self.assertEqual(batch.class_id.ndim, 1)
                self.assertEqual(batch.image_crop.ndim, 4)
                self.assertEqual(batch.joint_weight.ndim, 3)
                self.assertEqual(batch.keypoints.ndim, 3)
                self.assertEqual(batch.keypoints_local.ndim, 3)
                self.assertEqual(batch.person_id.ndim, 1)
                self.assertEqual(batch.class_id.ndim, 1)

                # check that the first dimension is B
                self.assertEqual(batch.class_id.size(0), B)
                self.assertEqual(batch.image_crop.size(0), B)
                self.assertEqual(batch.joint_weight.size(0), B)
                self.assertEqual(batch.keypoints.size(0), B)
                self.assertEqual(batch.keypoints_local.size(0), B)
                self.assertEqual(batch.person_id.size(0), B)
                self.assertEqual(batch.class_id.size(0), B)

    def setUp(self):
        mkdir_if_missing(os.path.join(PROJECT_ROOT, "./tests/test_data/TEST_ds/"))

    def tearDown(self):
        dir_path = os.path.join(PROJECT_ROOT, "./tests/test_data/TEST_ds/")
        if is_abs_dir(dir_path):
            shutil.rmtree(dir_path)


class TestPoseTrack21ImageDataset(unittest.TestCase):
    def test_init_single(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        for path, lengths in [
            ("test_single_dataset_1", [1]),
            ("test_single_dataset_2", [2, 0, 3]),
        ]:
            with self.subTest(msg="path: {}, lengths: {}".format(path, lengths)):
                with HidePrint():
                    ds = PoseTrack21_Image(config=cfg, path=[path])
                for i, length in enumerate(lengths):
                    r = ds[i]
                    self.assertTrue(isinstance(r, State))
                    self.assertEqual(len(r), length)
                    self.assertEqual(r.image_crop.size(0), length)

    def test_init_multiple(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        for path, lengths in [
            ("test_multi_dataset", [1, 2, 0, 3, 2, 0, 3]),
            ("test_directory_dataset", [1, 2, 0, 3]),
        ]:
            with self.subTest(msg="path: {}, lengths: {}".format(path, lengths)):
                with HidePrint():
                    ds = get_pose_track_21(config=cfg, path=[path], ds_name="image")
                self.assertEqual(len(ds), len(lengths))

                for i, length in enumerate(lengths):
                    r = ds[i]
                    self.assertTrue(isinstance(r, State))
                    self.assertEqual(len(r), length, f"r: {r}, i: {i}, len: {length}")
                    self.assertEqual(r.image_crop.size(0), length)

    def test_dataloader(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")
        lengths = [1, 2, 0, 3]
        with HidePrint():
            dl = get_data_loader(config=cfg, path=["test_dataloader_img"])
        self.assertEqual(len(dl), len(lengths))

        batch: State

        for i, batch in enumerate(dl):
            with self.subTest(msg=f"i: {i}, batch: {batch}"):

                B = lengths[i]
                self.assertTrue(isinstance(batch, State))
                self.assertEqual(len(batch), B)

                if B == 0:
                    self.assertEqual(batch, EMPTY_STATE)
                    for k in ["image_crop", "joint_weight", "keypoints", "keypoints_local", "person_id", "class_id"]:
                        self.assertTrue(k not in batch)
                    continue

                # check the number of dimensions
                self.assertEqual(batch.class_id.ndim, 1)
                self.assertTrue(all(img.ndim == 4 for img in batch.image))
                self.assertEqual(batch.image_crop.ndim, 4)
                self.assertEqual(batch.joint_weight.ndim, 3)
                self.assertEqual(batch.keypoints.ndim, 3)
                self.assertEqual(batch.keypoints_local.ndim, 3)
                self.assertEqual(batch.person_id.ndim, 1)
                self.assertEqual(batch.class_id.ndim, 1)

                # check that the first dimension is B
                self.assertEqual(batch.class_id.size(0), B)
                self.assertEqual(len(batch.image), max(B, 1))  # The frame-based dataset adds 1 img path nevertheless
                self.assertEqual(batch.image_crop.size(0), B)
                self.assertEqual(batch.joint_weight.size(0), B)
                self.assertEqual(batch.keypoints.size(0), B)
                self.assertEqual(batch.keypoints_local.size(0), B)
                self.assertEqual(batch.person_id.size(0), B)
                self.assertEqual(batch.class_id.size(0), B)

    def setUp(self):
        mkdir_if_missing(os.path.join(PROJECT_ROOT, "./tests/test_data/TEST_ds/"))

    def tearDown(self):
        dir_path = os.path.join(PROJECT_ROOT, "./tests/test_data/TEST_ds/")
        if is_abs_dir(dir_path):
            shutil.rmtree(dir_path)


if __name__ == "__main__":
    unittest.main()
