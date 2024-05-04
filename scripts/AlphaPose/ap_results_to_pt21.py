import glob
import os

from tqdm import tqdm

from dgs.utils.constants import PROJECT_ROOT
from dgs.utils.files import is_abs_file, read_json, write_json
from dgs.utils.types import FilePath

AP_JSON_FILES: FilePath = "./results/ap_results/FastPose/*/alphapose-results.json"
GT_DATA_BASE_PATH: FilePath = "./data/PoseTrack21/posetrack_data/val/"
OUTFILE_NAME: str = "pt21_results.json"

if __name__ == "__main__":
    ap_files = glob.glob(os.path.join(PROJECT_ROOT, AP_JSON_FILES))

    for ap_file in tqdm(ap_files, total=len(ap_files)):

        ap_data: list[dict[str, any]] = read_json(ap_file)
        ap_base_path, dataset_name = os.path.split(os.path.dirname(ap_file))
        new_filename: FilePath = os.path.join(ap_base_path, f"./files/{dataset_name}.json")

        if is_abs_file(new_filename):
            print(f"skipped file due to existence: {new_filename}")
            continue

        # get the gt data
        gt_data: dict = read_json(f"{GT_DATA_BASE_PATH}{dataset_name}.json")
        # get all the gt images
        gt_images: list[dict[str, any]] = gt_data["images"]

        for i in range(len(ap_data)):
            # rename box to bbox
            ap_data[i]["bbox"] = ap_data[i]["box"]
            # rename score to scores
            ap_data[i]["scores"] = [ap_data[i]["score"] for _ in range(17)]
            # rename idx to track_idx
            ap_data[i]["track_id"] = ap_data[i]["idx"]
            # replace ap_img_id with respective gt_img_id
            ap_data[i]["image_id"] = gt_images[ap_data[i]["image_id"]]["image_id"]

        new_json_data: dict[str, list[dict[str, any]]] = {
            "images": gt_images,  # re-add GT image information
            "annotations": ap_data,
            "categories": gt_data["categories"],  # re-add GT categories
        }

        write_json(new_json_data, new_filename)
