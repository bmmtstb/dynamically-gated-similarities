import unittest

from dgs.utils.image import load_image, load_video


class TestImage(unittest.TestCase):
    def test_load_image(self):
        for fp, shape in [
            ("./tests/test_data/283-200x300.jpg", (3, 300, 200)),
            ("./tests/test_data/file_example_PNG_500kB.png", (3, 566, 850)),
        ]:
            with self.subTest(msg=f"image name: {fp}"):
                self.assertEqual(load_image(fp).shape, shape)


class TestVideo(unittest.TestCase):
    def test_load_video(self):
        for fp, shape in [
            ("./tests/test_data/file_example_AVI_640_800kB.avi", (901, 3, 360, 640)),
        ]:
            with self.subTest(msg=f"image name: {fp}"):
                self.assertEqual(load_video(fp).shape, shape)


if __name__ == "__main__":
    unittest.main()
