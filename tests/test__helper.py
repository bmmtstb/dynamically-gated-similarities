import unittest

from dgs.utils.types import Device
from tests.helper import load_test_image, load_test_images, test_multiple_devices

all_images: list[str] = [
    "866-200x300.jpg",
    "866-200x800.jpg",
    "866-300x200.jpg",
    "866-500x500.jpg",
    "866-500x1000.jpg",
    "866-800x200.jpg",
    "866-1000x500.jpg",
    "866-1000x1000.jpg",
    "file_example_PNG_500kB.png",
]


class TestTestHelpers(unittest.TestCase):
    device_id: int = 0
    devices: list[Device] = ["cpu", "cuda"]

    @test_multiple_devices
    def test_test_multiple_devices(self, device):
        self.assertEqual(device.type, self.devices[self.device_id])
        self.device_id += 1

    def test_load_test_image(self):
        for w, h in [
            (200, 300),
            (200, 800),
            (300, 200),
            (500, 500),
            (500, 1000),
            (800, 200),
            (1000, 500),
            (1000, 1000),
        ]:
            with self.subTest(msg=f"w,h"):
                fn: str = f"866-{w}x{h}.jpg"
                self.assertEqual(load_test_image(fn).shape, (1, 3, h, w))

    def test_load_test_images(self):
        for filenames, shape, force_reshape, kwargs in [
            (["866-200x300.jpg"], (1, 3, 300, 200), False, {}),
            (["866-200x300.jpg", "866-200x300.jpg", "866-200x300.jpg"], (3, 3, 300, 200), False, {}),
            (["866-200x300.jpg"], (1, 3, 100, 100), True, {"output_size": (100, 100)}),
            (all_images, (9, 3, 200, 200), True, {"mode": "inside-crop", "output_size": (200, 200)}),
        ]:
            with self.subTest(msg=f"filenames: {filenames}, shape: {shape}"):
                self.assertEqual(tuple(load_test_images(filenames, force_reshape, **kwargs).shape), shape)

    def test_load_test_images_exception(self):
        for filenames, exception in [
            (["866-200x300.jpg", "866-300x200.jpg"], ValueError),
            (all_images, ValueError),
        ]:
            with self.subTest(msg=f"filenames: {filenames}, exception: {exception}"):
                with self.assertRaises(exception):
                    load_test_images(filenames)


if __name__ == "__main__":
    unittest.main()
