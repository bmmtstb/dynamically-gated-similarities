"""Run the RCNN backbone over the whole dataset and save the results, so they don't have to be recomputed every time."""

import os.path
from glob import glob

import torch
from torchvision.io import write_jpeg
from torchvision.transforms.v2.functional import convert_image_dtype
from tqdm import tqdm

from dgs.models.loader import module_loader
from dgs.utils.config import DEF_VAL, load_config
from dgs.utils.files import mkdir_if_missing, read_json
from dgs.utils.state import State
from dgs.utils.types import Config, FilePath

CONFIG_FILE: str = "./configs/helpers/predict_rcnn.yaml"
DL_KEY: str = "RCNN_pt21_Backbone"
SUBM_KEY: str = "submission_pt21"

SCORE_THRESHS: list[float] = [0.85, 0.90, 0.95]
IOU_THRESHS: list[float] = [0.5, 0.6, 0.7, 0.8]

# IN images: "./data/PoseTrack21/crops/hxw/val/DATASET/*.jpg"
# OUT predictions: "./data/PoseTrack21/posetrack_data/rcnn_XXX_YYY_val/DATASET.json"
# OUT cropped image and loc_kp: "./data/PoseTrack21/crops/hxw/rcnn_XXX_YYY_val/DATASET/*.jpg" and "...*.pt"


def save_crops(_s: State, _img_path: FilePath, _gt_img_id: str | int) -> None:
    for i in range(_s.B):
        img_path = os.path.join(_img_path, f"{str(_gt_img_id)}_{_s['person_id'][i]}.jpg")
        if os.path.exists(img_path):
            continue
        write_jpeg(
            input=convert_image_dtype(_s.image_crop[i], torch.uint8).cpu(),
            filename=img_path,
            quality=DEF_VAL["images"]["jpeg_quality"],
        )
        torch.save(_s.keypoints_local[i].unsqueeze(0).cpu(), str(img_path).replace(".jpg", ".pt"))


if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}")

    config: Config = load_config(CONFIG_FILE)

    h, w = config[DL_KEY]["crop_size"]

    for score_threshold in (pbar_score_thresh := tqdm(SCORE_THRESHS, desc="Score-Threshold")):
        pbar_score_thresh.set_postfix_str(str(score_threshold))

        score_str = f"{int(score_threshold * 100):03d}"
        config[DL_KEY]["score_threshold"] = score_threshold

        for iou_threshold in (pbar_iou_thresh := tqdm(IOU_THRESHS, desc="IoU-Threshold")):
            pbar_iou_thresh.set_postfix_str(str(iou_threshold))

            iou_str = f"{int(iou_threshold * 100):03d}"
            config[DL_KEY]["iou_threshold"] = iou_threshold

            dataset_paths: list = glob(config[DL_KEY]["dataset_paths"])
            rcnn_cfg_str: str = f"rcnn_{score_str}_{iou_str}_val"

            for dataset_path in (pbar_dataset := tqdm(dataset_paths, desc="datasets", leave=False)):
                ds_name = os.path.basename(os.path.realpath(dataset_path))
                pbar_dataset.set_postfix_str(ds_name)

                # GT data
                gt_data_path = f"./data/PoseTrack21/posetrack_data/val/{ds_name}.json"
                gt_data = read_json(gt_data_path)
                gt_img_id_map = [img["image_id"] for img in gt_data["images"]]  # zero-indexed!

                # modify the configuration
                config[DL_KEY]["data_path"] = dataset_path
                config[SUBM_KEY]["file"] = f"./data/PoseTrack21/posetrack_data/{rcnn_cfg_str}/{ds_name}.json"
                config[DL_KEY]["mask_path"] = gt_data_path

                if os.path.exists(config["submission"]["file"]):
                    continue

                dl_module = module_loader(config=config, module_class="dataloader", key=DL_KEY)
                subm_module = module_loader(config=config, module_class="submission", key="submission")

                # create img output folder
                crops_folder = f"./data/PoseTrack21/crops/{h}x{w}/{rcnn_cfg_str}/{ds_name}/"
                mkdir_if_missing(crops_folder)

                batch: list[State]
                for batch in tqdm(dl_module, desc="batch", leave=False):

                    for s in batch:
                        # images
                        # {
                        #     "is_labeled": true,
                        #     "nframes": 99,
                        #     "image_id": 10005830000,
                        #     "id": 10005830000,
                        #     "vid_id": "000583",
                        #     "file_name": "images/val/000583_mpii_test/000000.jpg",
                        #     "has_labeled_person": true,
                        #     "ignore_regions_y": [],
                        #     "ignore_regions_x": []
                        # }

                        # validate image data
                        _ = subm_module.get_image_data(s)

                        assert all(s.filepath[0] == s.filepath[i] for i in range(len(s.filepath)))
                        fp = "/".join(s.filepath[0].split("/")[-4:])

                        B = s.B

                        own_iid = int(s["image_id"][0].item())  # one-indexed
                        gt_img_id: int = gt_img_id_map[own_iid - 1]

                        # create dict with all the image data
                        image = {
                            "is_labeled": True,
                            "nframes": gt_data["images"][0]["nframes"],
                            "image_id": gt_img_id,
                            "id": gt_img_id,
                            "vid_id": gt_data["images"][0]["vid_id"],
                            "file_name": fp,
                            "has_labeled_person": B != 0,
                            "ignore_regions_x": gt_data["images"][own_iid - 1]["ignore_regions_x"],
                            "ignore_regions_y": gt_data["images"][own_iid - 1]["ignore_regions_y"],
                        }

                        subm_module.data["images"].append(image)

                        if B == 0:
                            continue

                        # annotations
                        # {
                        #     "bbox": [],
                        #     "bbox_head": [],
                        #     "category_id": 1,
                        #     "id": 1000583000000,
                        #     "image_id": 10005830000,
                        #     "keypoints": [],
                        #     "person_id": 38,
                        #     "track_id": 1
                        # }
                        s["pred_tid"] = (
                            torch.ones_like(s.person_id, dtype=torch.long, device=s.device) * -1
                        )  # set to -1
                        # set to number to remove duplicates in crop files
                        s["person_id"] = torch.arange(start=1, end=B + 1, dtype=torch.long, device=s.device)

                        annos = subm_module.get_anno_data(s)
                        for a_i, anno in enumerate(annos):
                            annos[a_i]["bbox"] = anno["bboxes"]
                            annos[a_i]["bbox_head"] = anno["bboxes"]
                            annos[a_i]["image_id"] = gt_img_id
                            annos[a_i]["id"] = gt_img_id
                            annos[a_i]["category_id"] = 1
                            # remove unnecessary keys
                            annos[a_i].pop("bboxes", None)
                            annos[a_i].pop("kps", None)

                        subm_module.data["annotations"] += annos
                        # save the image-crops and local key points
                        save_crops(s, _img_path=crops_folder, _gt_img_id=gt_img_id)
                subm_module.save()
