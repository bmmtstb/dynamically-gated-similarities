"""Given the json predictions by AlphaPose, convert them to the PT21 format."""

import glob
import os

from tqdm import tqdm

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import is_abs_file, read_json, write_json
from dgs.utils.types import FilePath

AP_BASE: FilePath = "./results/ap_results/"
AP_JSON_FILES: FilePath = "./*/alphapose-results.json"
GT_DATA_BASE_PATH: FilePath = "./data/PoseTrack21/posetrack_data/val/"
OUTFILE_NAME: str = "pt21_results.json"

FOLDERS: list[str] = [
    "FastPose",
    "HRNet",
    "SimpleBaseline",
    "DetectAndTrack_FastPose",
    "DetectAndTrack_HRNet",
    "DetectAndTrack_SimpleBaseline",
]

if __name__ == "__main__":
    for folder in FOLDERS:
        ap_files = glob.glob(os.path.join(PROJECT_ROOT, AP_BASE, f"./{folder}/", AP_JSON_FILES))

        for ap_file in tqdm(ap_files, total=len(ap_files)):

            ap_base_path, dataset_name = os.path.split(os.path.dirname(ap_file))
            new_filename: FilePath = os.path.join(ap_base_path, f"./results_json/{dataset_name}.json")

            if is_abs_file(new_filename):
                # print(f"skipped file due to existence: {new_filename}")
                continue

            ap_data: list[dict[str, any]] = read_json(ap_file)

            # get the gt data
            gt_data: dict = read_json(f"{GT_DATA_BASE_PATH}{dataset_name}.json")
            # get all the gt images
            gt_images: list[dict[str, any]] = gt_data["images"]

            for i, apdi in enumerate(ap_data):
                # rename box to bbox
                ap_data[i]["bbox"] = apdi["box"]

                # use AlphaPoses joint confidence score as score
                ap_data[i]["scores"] = apdi["keypoints"][2::3]

                # rename idx to track_idx (always taking the first value if multiple are present)
                if isinstance(apdi["idx"], list):
                    assert len(apdi["idx"]) != 0
                    ap_data[i]["track_id"] = int(
                        apdi["idx"][0][0] if isinstance(apdi["idx"][0], list) else apdi["idx"][0]
                    )
                else:
                    ap_data[i]["track_id"] = int(apdi["idx"])

                # replace ap_img_id with respective gt_img_id
                ap_data[i]["image_id"] = gt_images[apdi["image_id"]]["image_id"]

            new_json_data: dict[str, list[dict[str, any]]] = {
                "images": gt_images,  # re-add GT image information
                "annotations": ap_data,
                "categories": gt_data["categories"],  # re-add GT categories
            }

            write_json(new_json_data, new_filename)
