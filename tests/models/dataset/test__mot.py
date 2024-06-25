import os
import shutil
import unittest

from dgs.models.dataset.MOT import load_seq_ini, MOTImage, write_seq_ini
from dgs.models.loader import get_data_loader
from dgs.utils.config import load_config
from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.exceptions import InvalidPathException
from dgs.utils.files import is_abs_dir, mkdir_if_missing
from dgs.utils.state import State
from dgs.utils.utils import HidePrint


class TestSeqinfoIni(unittest.TestCase):

    seqinfo_path = "./tests/test_data/MOT_test/seqinfo.ini"
    new_path = "./tests/test_data/MOT_test/seqinfo_test.ini"

    test_data = {
        "name": "MOT_test",
        "imDir": "img1",
        "frameRate": "42",
        "seqLength": "3",
        "imWidth": "480",
        "imHeight": "640",
        "imExt": ".jpg",
    }

    def test_load_seq_ini(self):
        data = load_seq_ini(self.seqinfo_path)
        self.assertTrue(isinstance(data, dict))

        for key in data.keys():
            with self.subTest(msg="key: {} is string".format(key)):
                self.assertTrue(isinstance(key, str))

        for key, val in self.test_data.items():
            with self.subTest(msg="key: {}, val: {}".format(key, val)):
                self.assertTrue(key in data, data)
                self.assertEqual(data[key], val, data)

    def test_load_seq_ini_key(self):
        data = load_seq_ini(self.seqinfo_path, key="Other")
        self.assertEqual(len(data), 1)
        self.assertTrue("name" in data)
        self.assertEqual(data["name"], "MOT_other")

    def test_load_seq_ini_exceptions(self):
        with self.assertRaises(InvalidPathException) as e:
            _ = load_seq_ini("./tests/test_data/seq.info")
        self.assertTrue("file './tests/test_data/seq.info' does not have .ini" in str(e.exception), msg=e.exception)

        with self.assertRaises(KeyError) as e:
            _ = load_seq_ini(fp=self.seqinfo_path, key="Dummy")
        self.assertTrue("Expected key 'Dummy' to be in" in str(e.exception), msg=e.exception)

    def test_write_seq_ini(self):
        for space in [None, True, False]:
            with self.subTest(msg="space: {}".format(space)):
                self.assertFalse(os.path.exists(self.new_path))
                write_seq_ini(fp=self.new_path, data=self.test_data, space_around_delimiters=space)
                self.assertTrue(os.path.exists(self.new_path))
                self.assertDictEqual(load_seq_ini(self.new_path), self.test_data)
                os.remove(self.new_path)

    def test_override_seq_ini_key(self):
        key = "Other"
        self.assertFalse(os.path.exists(self.new_path))
        write_seq_ini(fp=self.new_path, data=self.test_data, key=key)
        self.assertTrue(os.path.exists(self.new_path))
        self.assertDictEqual(load_seq_ini(self.new_path, key=key), self.test_data)
        other_data = self.test_data.copy()
        other_data["dummy"] = "dummy"
        write_seq_ini(fp=self.new_path, data=other_data, key=key)
        self.assertTrue(os.path.exists(self.new_path))
        self.assertDictEqual(load_seq_ini(self.new_path, key=key), other_data)
        os.remove(self.new_path)

    def test_add_seq_ini_key(self):
        key1 = "Existing"
        key2 = "Other"
        other_data = self.test_data.copy()
        other_data["dummy"] = "dummy"
        self.assertFalse(os.path.exists(self.new_path))
        write_seq_ini(fp=self.new_path, data=self.test_data, key=key1)
        write_seq_ini(fp=self.new_path, data=other_data, key=key2)
        self.assertTrue(os.path.exists(self.new_path))
        self.assertDictEqual(load_seq_ini(self.new_path, key=key1), self.test_data)
        self.assertDictEqual(load_seq_ini(self.new_path, key=key2), other_data)
        os.remove(self.new_path)

    def test_write_seq_ini_exceptions(self):
        self.assertFalse(os.path.exists(self.new_path))
        for missing in self.test_data.keys():
            with self.subTest(msg="missing: {}".format(missing)):
                data = self.test_data.copy()
                data.pop(missing)
                with self.assertRaises(ValueError) as e:
                    write_seq_ini(fp=self.new_path, data=data)
                self.assertTrue(f"Expected '{missing}' to be in data, but got" in str(e.exception), msg=e.exception)
                self.assertFalse(os.path.exists(self.new_path))

    def setUp(self):
        if os.path.exists(self.new_path):
            os.remove(self.new_path)

    def tearDown(self):
        if os.path.exists(self.new_path):
            os.remove(self.new_path)


class TestMOTImageDataset(unittest.TestCase):
    seqinfo_path = "./tests/test_data/seqinfo.ini"
    test_cfg = load_config("./tests/test_data/configs/test_config_MOT.yaml")
    logging_path = os.path.join(PROJECT_ROOT, "./tests/test_data/TEST_ds/")

    def test_MOTImage_dataset(self):
        for path, lengths in [
            ("test_single_dataset_1", [2, 1, 0]),
            ("test_single_dataset_2", [2, 1, 0]),
        ]:
            with self.subTest(msg="path: {}, lengths: {}".format(path, lengths)):
                with HidePrint():
                    ds = MOTImage(config=self.test_cfg.copy(), path=[path])
                self.assertEqual(len(ds), len(lengths))

                for i, length in enumerate(lengths):
                    r = ds[i]
                    self.assertTrue(isinstance(r, State))
                    self.assertEqual(len(r), length, f"r: {r}, i: {i}, len: {length}")
                    self.assertEqual(r.image_crop.size(0), length)

    def test_MOTImage_dataloader(self):
        lengths = [2, 1, 0]

        for name, batch_sizes in [
            ("test_dataloader_img", [1, 1, 1]),
            ("test_dataloader_img_batched", [2, 1]),
        ]:
            with HidePrint():
                dl = get_data_loader(config=self.test_cfg.copy(), path=[name])

            batch: list[State]
            for i, batch in enumerate(dl):
                with self.subTest(msg=f"name: {name}, i: {i}, batch: {batch}"):
                    self.assertTrue(isinstance(batch, list))
                    self.assertEqual(len(batch), batch_sizes[i])
                    for j, sub_state in enumerate(batch):
                        self.assertEqual(len(sub_state), lengths[i * batch_sizes[0] + j])

    def setUp(self):
        mkdir_if_missing(self.logging_path)

    def tearDown(self):
        if is_abs_dir(self.logging_path):
            shutil.rmtree(self.logging_path)


if __name__ == "__main__":
    unittest.main()
