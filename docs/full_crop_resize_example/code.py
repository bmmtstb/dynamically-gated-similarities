import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as tvt
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes, draw_keypoints, make_grid

from dgs.utils.constants import SKELETONS
from dgs.utils.files import read_json
from dgs.utils.image import CustomCropResize, CustomToAspect, load_image
from dgs.utils.validation import validate_bboxes, validate_images, validate_key_points
from dgs.utils.visualization import torch_show_image


def transform_crop_resize() -> tvt.Compose:
    return tvt.Compose(
        [
            tvt.ConvertBoundingBoxFormat(format=tv_tensors.BoundingBoxFormat.XYWH),
            tvt.ClampBoundingBoxes(),  # make sure the bboxes are clamped to start with
            CustomCropResize(),  # crop the image at the four corners specified in bboxes
            tvt.ClampBoundingBoxes(),  # duplicate ?
        ]
    )


if __name__ == "__main__":
    # load image and key point predictions
    img = load_image("./docs/full_crop_resize_example/1_original.jpg")
    json = read_json("./docs/full_crop_resize_example/alphapose-results.json")

    H, W = img.shape[-2:]
    J = len(json)

    # set up bounding boxes and key-point coordinates
    bboxes = validate_bboxes(tv_tensors.BoundingBoxes([det["box"] for det in json], canvas_size=(H, W), format="XYWH"))
    coords = validate_key_points(torch.tensor([det["keypoints"] for det in json]).float().reshape(J, -1, 3))[:, :, :2]

    # plot and save image with bounding boxes
    torch_show_image(draw_bounding_boxes(img, box_convert(bboxes.detach().clone(), "xywh", "xyxy")))
    plt.savefig(fname=f"./docs/full_crop_resize_example/2_bboxes.jpg", format="jpg", dpi=300)

    for i, mode in enumerate(CustomToAspect.modes):
        structured_input = {
            "image": validate_images(load_image("./docs/full_crop_resize_example/1_original.jpg")),
            "box": validate_bboxes(bboxes),
            "keypoints": validate_key_points(coords),
            "output_size": (200, 200),
            "mode": mode,
            "fill": (255, 0, 255),  # pink for fill pad
        }
        r = transform_crop_resize()(structured_input)
        new_images = r["image"]
        new_kp = r["keypoints"]
        torch_show_image(
            make_grid(
                [
                    draw_keypoints(
                        ni, validate_key_points(nkp), connectivity=SKELETONS["coco"], colors="blue", radius=4
                    )
                    for ni, nkp in zip(new_images, new_kp)
                ],
                nrow=4,
            )
        )
        plt.savefig(
            fname=f"./docs/full_crop_resize_example/3_{i}_person_crops_{mode}.jpg",
            format="jpg",
            dpi=300,
            bbox_inches="tight",
        )
