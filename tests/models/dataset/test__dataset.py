import unittest
from unittest.mock import patch

from dgs.models.dataset import DATASETS, get_dataset, register_dataset
from dgs.models.dataset.dataset import BaseDataset
from dgs.models.dataset.posetrack21 import PoseTrack21_BBox


class TestBaseDataset(unittest.TestCase):

    def test_get(self):
        ds = get_dataset("PoseTrack21JSON")

        self.assertTrue(isinstance(ds, type))
        self.assertTrue(issubclass(ds, BaseDataset))

    def test_register(self):
        with patch.dict(DATASETS):
            for name, func, exception in [
                ("dummy", PoseTrack21_BBox, False),
                ("dummy", PoseTrack21_BBox, KeyError),
                ("new_dummy", PoseTrack21_BBox, False),
            ]:
                with self.subTest(msg="name: {}, func: {}, except: {}".format(name, func, exception)):
                    if exception is not False:
                        with self.assertRaises(exception):
                            register_dataset(name, func)
                    else:
                        register_dataset(name, func)
                        self.assertTrue("dummy" in DATASETS)
        self.assertTrue("dummy" not in DATASETS)
        self.assertTrue("new_dummy" not in DATASETS)


if __name__ == "__main__":
    unittest.main()
