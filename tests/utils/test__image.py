import os.path
import unittest

import imagesize

from dgs.utils.files import project_to_abspath
from dgs.utils.image import compute_padding, load_image, load_video

# Map image name to shape.
# Shape is torch shape and therefore [C x H x W].
TEST_IMAGES: dict[str, tuple[int, int, int]] = {
    "866-200x300.jpg": (3, 300, 200),
    "866-200x800.jpg": (3, 800, 200),
    "866-300x200.jpg": (3, 200, 300),
    "866-500x500.jpg": (3, 500, 500),
    "866-500x1000.jpg": (3, 1000, 500),
    "866-800x200.jpg": (3, 200, 800),
    "866-1000x500.jpg": (3, 500, 1000),
    "866-1000x1000.jpg": (3, 1000, 1000),
    "file_example_PNG_500kB.png": (3, 566, 850),
}


class TestImageUtils(unittest.TestCase):
    def test_compute_padding(self):
        for width, height, target_aspect, paddings, msg in [
            # paddings is LTRB
            (11, 11, 2, [5, 0, 6, 0], "square to landscape"),
            (11, 11, 0.5, [0, 5, 0, 6], "square to portrait"),
            (11, 11, 1, [0, 0, 0, 0], "square to square - zeros"),
            (200, 50, 1, [0, 75, 0, 75], "landscape to square"),
            (50, 75, 1, [12, 0, 13, 0], "portrait to square"),
            (200, 100, 4, [100, 0, 100, 0], "landscape to wider landscape"),
            (400, 100, 2, [0, 50, 0, 50], "landscape to slim landscape"),
            (100, 200, 1 / 4, [0, 100, 0, 100], "portrait to long portrait"),
            (50, 75, 5 / 6, [6, 0, 6, 0], "portrait to short portrait"),  # (w=62.5,h=75)
            (50, 100, 2, [75, 0, 75, 0], "portrait to landscape"),
            (150, 75, 0.5, [0, 112, 0, 113], "landscape to portrait"),
        ]:
            with self.subTest(msg=msg):
                self.assertListEqual(compute_padding(width, height, target_aspect), paddings)


class TestImage(unittest.TestCase):
    def test_load_image(self):
        for file_name, shape in TEST_IMAGES.items():
            with self.subTest(msg=f"image name: {file_name}"):
                fp = project_to_abspath(os.path.join("./tests/test_data/", file_name))
                self.assertEqual(load_image(fp).shape, shape)
                self.assertEqual(imagesize.get(fp), shape[-1:-3:-1])


class TestVideo(unittest.TestCase):
    def test_load_video(self):
        for fp, shape in [
            ("./tests/test_data/file_example_AVI_640_800kB.avi", (901, 3, 360, 640)),
        ]:
            with self.subTest(msg=f"image name: {fp}"):
                self.assertEqual(load_video(fp).shape, shape)


if __name__ == "__main__":
    unittest.main()
