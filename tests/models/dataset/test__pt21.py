import unittest

from dgs.models.dataset.posetrack21 import get_pose_track_21, PoseTrack21_BBox, PoseTrack21_Image, validate_pt21_json
from dgs.models.loader import get_data_loader
from dgs.utils.config import load_config
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

    def test_get_items(self):
        cfg = load_config("./tests/test_data/configs/test_config_pt21.yaml")

        with HidePrint():
            dl = get_data_loader(config=cfg, path=["test_dataloader"])
        for batch in dl:
            self.assertTrue(isinstance(batch, State))
            self.assertEqual(len(batch), int(cfg["test_dataloader"]["batch_size"]))


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


if __name__ == "__main__":
    unittest.main()
