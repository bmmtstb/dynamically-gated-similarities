"""Run the RCNN backbone over the whole dataset and save the results, so they don't have to be recomputed every time."""

import os.path
import time
import warnings
from glob import glob

import torch as t
from torchvision.io import write_jpeg
from torchvision.transforms.v2.functional import convert_image_dtype
from tqdm import tqdm

from dgs.models.loader import module_loader
from dgs.models.submission import PoseTrack21Submission
from dgs.utils.config import DEF_VAL, load_config
from dgs.utils.files import mkdir_if_missing, read_json
from dgs.utils.state import State
from dgs.utils.types import Config, FilePath
from dgs.utils.utils import HidePrint, replace_file_type, send_discord_notification

CONFIG_FILE: str = "./configs/helpers/predict_rcnn.yaml"
SUBM_KEY: str = "submission_pt21"

DL_KEYS: list[str] = [
    "PT21_256x128_val",
    "PT21_256x192_val",
    "PT21_256x192_train",
]

RCNN_DL_KEYS: dict[str, tuple[list[float], list[float]]] = {
    "RCNN_PT21_256x128_val": ([0.80, 0.85, 0.90, 0.95], [0.3, 0.4, 0.5, 0.6]),
    "RCNN_PT21_256x192_val": ([0.75, 0.80, 0.85, 0.90, 0.95, 0.99], [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]),
    "RCNN_PT21_256x192_train": ([0.85], [0.4]),
    "RCNN_PT21_256x192_test": ([0.85], [0.4]),
}

# IN images: "./data/PoseTrack21/images/{val|train}/DATASET/*.jpg"
# OUT predictions: "./data/PoseTrack21/posetrack_data/rcnn_XXX_YYY_{val|train}/DATASET.json"
# OUT cropped image and loc_kp: "./data/PoseTrack21/crops/{h}x{w}/rcnn_XXX_YYY_{val|train}/DATASET/*.jpg" and "...*.pt"


def save_crops(_s: State, img_dir: FilePath, _gt_img_id: str | int) -> None:
    """Save the image crops and local key points."""
    for i in range(_s.B):
        assert t.all(_s["person_id"] >= 0)
        img_path = os.path.join(img_dir, f"{str(_gt_img_id)}_{_s['person_id'][i]}.jpg")
        if os.path.exists(img_path):
            continue
        if "image_crop" not in _s or "keypoints_local" not in _s:
            _s.load_image_crop(store=True)
        write_jpeg(
            input=convert_image_dtype(_s.image_crop[i], t.uint8).cpu(),
            filename=img_path,
            quality=DEF_VAL["images"]["jpeg_quality"],
        )
        if "joint_weight" in _s:
            weights = _s.joint_weight[i].unsqueeze(0).cpu()
        else:
            weights = t.ones((1, _s.J, 1), dtype=t.float32)
        kp_loc = t.cat([_s.keypoints_local[i].unsqueeze(0).cpu(), weights], dim=-1)
        kp_glob = t.cat([_s.keypoints[i].unsqueeze(0).cpu(), weights], dim=-1)
        t.save(kp_loc, replace_file_type(str(img_path), new_type=".pt"))
        t.save(kp_glob, replace_file_type(str(img_path), new_type="_glob.pt"))


def predict_and_save_rcnn(config: Config, dl_key: str, subm_key: str, rcnn_cfg_str: str) -> None:
    """Predict and save the rcnn results of all the PT21 datasets in the folder given by the config."""
    # pylint: disable=too-many-locals

    dataset_paths: list = glob(config[dl_key]["dataset_paths"])

    for dataset_path in (pbar_dataset := tqdm(dataset_paths, desc="datasets", leave=False)):
        # dataset_path: "./data/PoseTrack21/images/{val|train}/DATASET/"
        ds_name = os.path.basename(os.path.realpath(dataset_path))
        pbar_dataset.set_postfix_str(ds_name)

        gt_data_path = f"{dataset_path.replace('images', 'posetrack_data').rstrip('/')}.json"
        if not os.path.exists(gt_data_path):
            warnings.warn(f"Could not find the ground-truth data at: {gt_data_path}")
            continue
        gt_imgs = read_json(gt_data_path)["images"]
        gt_img_id_map = [img["image_id"] for img in gt_imgs]  # zero-indexed!

        # create img output folder
        crop_h, crop_w = config[dl_key]["crop_size"]
        crops_folder = f"./data/PoseTrack21/crops/{crop_h}x{crop_w}/{rcnn_cfg_str}/{ds_name}/"
        mkdir_if_missing(crops_folder)

        # modify the configuration
        config[dl_key]["data_path"] = dataset_path
        config[dl_key]["mask_path"] = gt_data_path
        config[subm_key]["file"] = f"./data/PoseTrack21/posetrack_data/{crop_h}x{crop_w}_{rcnn_cfg_str}/{ds_name}.json"

        if os.path.exists(config[subm_key]["file"]):
            continue

        dl_module = module_loader(config=config, module_type="dataloader", key=dl_key)
        subm_module: PoseTrack21Submission = module_loader(config=config, module_type="submission", key=subm_key)

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
                fp = "/".join(s.filepath[0].split("/")[-4:])  # get fp in dataset folder

                own_iid = int(s["image_id"][0].item())  # one-indexed
                gt_img_id: int = gt_img_id_map[own_iid - 1]

                # create dict with all the image data and append it to the submission module
                subm_module.data["images"].append(
                    {
                        "is_labeled": True,
                        "nframes": gt_imgs[0]["nframes"],
                        "image_id": gt_img_id,
                        "id": gt_img_id,
                        "vid_id": gt_imgs[0]["vid_id"],
                        "file_name": fp,
                        "has_labeled_person": s.B != 0,
                        "ignore_regions_x": gt_imgs[own_iid - 1]["ignore_regions_x"],
                        "ignore_regions_y": gt_imgs[own_iid - 1]["ignore_regions_y"],
                    }
                )

                if s.B == 0:
                    continue

                # annotations
                # {
                #     "bbox": [],
                #     "bbox_head": [],
                #     "category_id": 1,
                #     "id": 1000583000000,
                #     "image_id": 10005830000,
                #     "keypoints": [],
                #     "track_id": 1
                # }
                s["pred_tid"] = t.ones_like(s.person_id, dtype=t.long, device=s.device) * -1  # set to -1
                # set to number to remove duplicates in crop files
                s["person_id"] = t.arange(start=1, end=s.B + 1, dtype=t.long, device=s.device)

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
                save_crops(s, img_dir=crops_folder, _gt_img_id=gt_img_id)
        subm_module.save()


def extract_gt_boxes(config: Config, dl_key: str) -> None:
    """Given the gt annotations, extract the image crops and local coordinates."""
    dataset_paths: list[str] = glob(config[dl_key]["dataset_paths"])

    for dataset_path in (pbar_dataset := tqdm(dataset_paths, desc="datasets", leave=False)):
        ds_name = os.path.basename(os.path.realpath(dataset_path))
        pbar_dataset.set_postfix_str(ds_name)

        # create img output folder
        crop_h, crop_w = config[dl_key]["crop_size"]
        crops_folder = (
            f"{config[dl_key]['dataset_path']}"
            f"./crops/{crop_h}x{crop_w}/{os.path.basename(os.path.dirname(dataset_path))}"
        )
        mkdir_if_missing(crops_folder)

        # modify the configuration
        config[dl_key]["data_path"] = dataset_path

        with HidePrint():
            dl_module = module_loader(config=config, module_type="dataloader", key=dl_key)

        batch: list[State]
        s: State
        for batch in tqdm(dl_module, desc="batch", leave=False):
            for s in batch:
                if s.B == 0:
                    continue
                # save the image-crops, there are no local key-points
                save_crops(s, img_dir=crops_folder, _gt_img_id=s["frame_id"][0].item())


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")

    start_time = time.time()

    print("Extracting the GT image crops")
    for DL_KEY in (pbar_dl := tqdm(DL_KEYS, desc="Dataloader", leave=False)):
        pbar_dl.set_postfix_str(DL_KEY)

        cfg: Config = load_config(CONFIG_FILE)
        extract_gt_boxes(config=cfg, dl_key=DL_KEY)

    print("Extracting using KeypointRCNN as detector")
    for RCNN_DL_KEY, (SCORE_THRESHS, IOU_THRESHS) in (
        pbar_dl := tqdm(RCNN_DL_KEYS.items(), desc="Dataloader", leave=False)
    ):
        pbar_dl.set_postfix_str(RCNN_DL_KEY)

        rcnn_cfg: Config = load_config(CONFIG_FILE)
        for score_threshold in (pbar_score_thresh := tqdm(SCORE_THRESHS, desc="Score-Threshold", leave=False)):
            pbar_score_thresh.set_postfix_str(str(score_threshold))

            score_str = f"{int(score_threshold * 100):03d}"
            rcnn_cfg[RCNN_DL_KEY]["score_threshold"] = score_threshold

            for iou_threshold in (pbar_iou_thresh := tqdm(IOU_THRESHS, desc="IoU-Threshold", leave=False)):
                pbar_iou_thresh.set_postfix_str(str(iou_threshold))

                iou_str = f"{int(iou_threshold * 100):03d}"
                rcnn_cfg[RCNN_DL_KEY]["iou_threshold"] = iou_threshold

                predict_and_save_rcnn(
                    config=rcnn_cfg,
                    dl_key=RCNN_DL_KEY,
                    subm_key=SUBM_KEY,
                    # append val / train / test to config string
                    rcnn_cfg_str=f"rcnn_{score_str}_{iou_str}_{RCNN_DL_KEY.rsplit('_', maxsplit=1)[-1]}",
                )

    if (elapsed_time := time.time() - start_time) > 300:  # 5 minutes
        send_discord_notification("finished extracting bboxes for PT21")
