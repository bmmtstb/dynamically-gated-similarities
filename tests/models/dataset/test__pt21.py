import os.path
import shutil
import unittest

from dgs.models.dataset.posetrack21 import get_pose_track_21, PoseTrack21_BBox, PoseTrack21_Image, validate_pt21_json
from dgs.models.loader import get_data_loader
from dgs.utils.config import load_config
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import is_abs_dir, mkdir_if_missing
from dgs.utils.state import State
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
                    self.assertEqual(len(r), length)
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

                if B != 0:
                    # check the number of dimensions
                    self.assertEqual(batch.class_id.ndim, 1)
                    self.assertEqual(batch.image.ndim, 4)
                    self.assertEqual(batch.image_crop.ndim, 4)
                    self.assertEqual(batch.joint_weight.ndim, 3)
                    self.assertEqual(batch.keypoints.ndim, 3)
                    self.assertEqual(batch.keypoints_local.ndim, 3)
                    self.assertEqual(batch.person_id.ndim, 1)
                    self.assertEqual(batch.class_id.ndim, 1)

                # check that the first dimension is B
                self.assertEqual(batch.class_id.size(0), B)
                self.assertEqual(batch.image.size(0), B)
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
