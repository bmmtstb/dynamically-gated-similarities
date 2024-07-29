"""Run the RCNN backbone over the whole dataset and save the results, so they don't have to be recomputed every time."""

import os
from glob import glob

import torch as t
from torchvision.io import write_jpeg
from torchvision.transforms.v2.functional import convert_image_dtype
from tqdm import tqdm

from dgs.models.dataset.MOT import load_seq_ini, write_seq_ini
from dgs.models.loader import module_loader
from dgs.models.submission import MOTSubmission
from dgs.utils.config import DEF_VAL, load_config
from dgs.utils.files import mkdir_if_missing
from dgs.utils.state import State
from dgs.utils.types import Config, FilePath
from dgs.utils.utils import notify_on_completion_or_error, send_discord_notification

CONFIG_FILE: str = "./configs/helpers/predict_rcnn.yaml"

SCORE_THRESHS: list[float] = [0.85, 0.90, 0.95, 0.99]
IOU_THRESHS: list[float] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

RCNN_DL_KEYS: list[str] = [
    # "RCNN_MOT_train",
    # "RCNN_MOT_test",
    # "RCNN_Dance_test",
    # "RCNN_Dance_train",
    "RCNN_Dance_val",
]

DL_KEYS: list[str] = [
    # "MOT_train",
    # "Dance_train",
    # "Dance_val",
]

# IN images: "./data/{MOT20|DanceTrack}/{train|test|val}/DATASET/img1/*.jpg"
# IN boxes: "./data/{MOT20|DanceTrack}/{train|test|val}/DATASET/gt/gt.txt"
# OUT predictions: "./data/{MOT20|DanceTrack}/{train|test|val}/DATASET/det/rcnn_XXX_YYY.txt"
# OUT cropped image and loc_kp: "./data/{MOT20|DanceTrack}/{train|test|val}/DATASET/rcnn_XXX_YYY/*.jpg" and "...*.pt"


@notify_on_completion_or_error(min_time=60)
@t.no_grad()
def run_RCNN_extractor(dl_key: str, subm_key: str, rcnn_cfg_str: str) -> None:
    """Given some configuration, predict and extract the image crops using Keypoint-RCNN."""
    dataset_paths: list = sorted(glob(config[dl_key]["dataset_paths"]))
    assert len(dataset_paths) > 0

    for dataset_path in (pbar_dataset := tqdm(dataset_paths, desc="datasets", leave=False)):
        dataset_path = os.path.normpath(dataset_path)
        pbar_dataset.set_postfix_str(os.path.basename(os.path.realpath(dataset_path)))

        # GT data
        gt_seqinfo = load_seq_ini(os.path.join(dataset_path, "./seqinfo.ini"), key="Sequence")

        # modify submission data - seqinfo.ini
        own_seqinfo = gt_seqinfo.copy()
        own_seqinfo["imDir"] = rcnn_cfg_str
        crop_h, crop_w = config[dl_key]["crop_size"]
        own_seqinfo["cropWidth"] = str(crop_w)
        own_seqinfo["cropHeight"] = str(crop_h)
        config[subm_key]["seqinfo_key"] = rcnn_cfg_str
        write_seq_ini(fp=os.path.join(dataset_path, "./seqinfo.ini"), data=own_seqinfo, key=rcnn_cfg_str)

        # modify the configuration
        config[dl_key]["data_path"] = os.path.normpath(os.path.join(dataset_path, f"./{gt_seqinfo['imDir']}/"))
        config[subm_key]["file"] = os.path.normpath(os.path.join(dataset_path, f"./det/{rcnn_cfg_str}.txt"))

        # create img output folder
        crops_folder = os.path.join(dataset_path, f"./{rcnn_cfg_str}/")
        mkdir_if_missing(crops_folder)

        if os.path.exists(config[subm_key]["file"]):
            # skip if submission file exists and there are as many detections in the crop folder as in the subm. file
            with open(config[subm_key]["file"], "r", encoding="utf-8") as subm_f:
                if (
                    len(subm_f.readlines())
                    == len(glob(crops_folder + "/*.jpg"))
                    == len(glob(crops_folder + "/*glob.pt"))
                ):
                    continue

        dataloader = module_loader(config=config, module_class="dataloader", key=dl_key)

        # load submission
        submission: MOTSubmission = module_loader(config=config, module_class="submission", key=subm_key)
        submission.seq_info = own_seqinfo

        batch: list[State]
        s: State
        frame_id = 0

        # get the amount of images per crop id
        all_crop_ids = {i: 0 for i in range(1, int(gt_seqinfo["seqLength"]) + 1)}
        for file_name in [os.path.basename(p) for p in glob(os.path.join(crops_folder, f"*{own_seqinfo['imExt']}"))]:
            file_name = file_name.rstrip(gt_seqinfo["imExt"])
            img_id, _ = file_name.split("_")
            all_crop_ids[int(img_id)] += 1

        assert len(dataloader) >= 0

        for batch in tqdm(dataloader, desc="batch", leave=False):
            for s in batch:
                frame_id += 1

                s["frame_id"] = t.tensor([frame_id] * s.B, dtype=t.long, device=s.device)
                s.person_id = t.arange(1, s.B + 1, dtype=t.long, device=s.device)
                s.track_id = t.arange(1, s.B + 1, dtype=t.long, device=s.device)
                # pred_tid is 0 indexed -> submission file will add 1 later
                s["pred_tid"] = t.arange(0, s.B, dtype=t.long, device=s.device)

                # append to submission
                submission.append(s)

                if s.B in (0, all_crop_ids[frame_id]):
                    s.clean()
                    continue

                # save the image-crops and local key points
                save_crops(
                    s,
                    img_dir=crops_folder,
                    _gt_img_id=frame_id,
                    save_kps=True,
                    crop_size=(crop_h, crop_w),
                    crop_mode="zero-pad",
                )
                assert tuple(s.image_crop.shape[-2:]) == (crop_h, crop_w)
                # remove image and image crop to free memory
                s.clean()
        submission.save()


@notify_on_completion_or_error(min_time=60)
@t.no_grad()
def run_gt_extractor(dl_key: str) -> None:
    """Given some ground-truth annotation data, extract and save the image crops"""
    dataset_paths: list = sorted(glob(config[dl_key]["dataset_paths"]))
    assert len(dataset_paths) > 0

    for dataset_path in (pbar_dataset := tqdm(dataset_paths, desc="datasets", leave=False)):
        ds_name = os.path.basename(os.path.realpath(dataset_path))
        pbar_dataset.set_postfix_str(ds_name)

        # GT data
        gt_seqinfo_path = os.path.join(dataset_path, "./seqinfo.ini")
        gt_seqinfo = load_seq_ini(gt_seqinfo_path, key="Sequence")

        # create img output folder
        crops_folder = os.path.normpath(os.path.join(dataset_path, "./crops/"))
        mkdir_if_missing(crops_folder)

        # modify submission data - seqinfo.ini
        own_seqinfo = gt_seqinfo.copy()
        own_seqinfo["imDir"] = "crops"
        crop_h, crop_w = config[dl_key]["crop_size"]
        own_seqinfo["cropWidth"] = str(crop_w)
        own_seqinfo["cropHeight"] = str(crop_h)
        write_seq_ini(fp=gt_seqinfo_path, data=own_seqinfo, key="Crops")

        # modify the configuration
        config[dl_key]["data_path"] = os.path.normpath(os.path.join(dataset_path, "./gt/gt.txt/"))

        # skip if at least one crop exists for every image in the sequence
        all_crop_ids = set(
            int(os.path.basename(p).split("_")[0]) for p in glob(os.path.join(crops_folder, f"*{own_seqinfo['imExt']}"))
        )
        if all(i in all_crop_ids for i in range(1, int(gt_seqinfo["seqLength"]) + 1)):  # 1 indexed
            continue

        # get data loader
        dataloader = module_loader(config=config, module_class="dataloader", key=dl_key)
        assert len(dataloader) >= 0

        batch: list[State]
        s: State

        for batch in tqdm(dataloader, desc="batch", leave=False):
            for s in batch:
                if s.B == 0 or len(glob(os.path.join(crops_folder, f"{s['frame_id'][0]}_*.jpg"))) == s.B:
                    s.clean()
                    continue
                # save the image-crops, there are no local key-points
                save_crops(
                    s,
                    img_dir=crops_folder,
                    _gt_img_id=s["frame_id"][0],
                    save_kps=False,
                    crop_size=(crop_h, crop_w),
                    crop_mode="zero-pad",
                )
                assert tuple(s.image_crop.shape[-2:]) == (crop_h, crop_w)

                # remove image and image crop to free memory
                s.clean()


@t.no_grad()
def save_crops(_s: State, img_dir: FilePath, _gt_img_id: str | int, save_kps: bool = True, **kwargs) -> None:
    """Save the image crops."""
    for i in range(_s.B):
        img_path = os.path.join(img_dir, f"{str(_gt_img_id)}_{_s['person_id'][i]}.jpg")
        if os.path.exists(img_path):
            continue
        if "image_crop" not in _s or (save_kps and "keypoints_local" not in _s):
            _s.load_image_crop(store=True, **kwargs)
        write_jpeg(
            input=convert_image_dtype(_s.image_crop[i], t.uint8).cpu(),
            filename=img_path,
            quality=DEF_VAL["images"]["jpeg_quality"],
        )
        if save_kps:
            t.save(_s.keypoints_local[i].unsqueeze(0).cpu(), str(img_path).replace(".jpg", ".pt"))
            t.save(_s.keypoints[i].unsqueeze(0).cpu(), str(img_path).replace(".jpg", "_glob.pt"))


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")

    config: Config = load_config(CONFIG_FILE)

    for DL_KEY in DL_KEYS:
        print(f"Extracting GT image crops using dataloader: {DL_KEY}")
        run_gt_extractor(dl_key=DL_KEY)

    for RCNN_DL_KEY in RCNN_DL_KEYS:
        print(f"Using Keypoint-RCNN to predict and extract crops for dataloader: {RCNN_DL_KEY}")

        h, w = config[RCNN_DL_KEY]["crop_size"]

        for score_threshold in (pbar_score_thresh := tqdm(SCORE_THRESHS, desc="Score-Threshold")):
            pbar_score_thresh.set_postfix_str(str(score_threshold))
            score_str = f"{int(score_threshold * 100):03d}"
            config[RCNN_DL_KEY]["score_threshold"] = score_threshold

            for iou_threshold in (pbar_iou_thresh := tqdm(IOU_THRESHS, desc="IoU-Threshold", leave=False)):
                pbar_iou_thresh.set_postfix_str(str(iou_threshold))
                iou_str = f"{int(iou_threshold * 100):03d}"

                config[RCNN_DL_KEY]["iou_threshold"] = iou_threshold

                _rcnn_cfg_str = f"rcnn_{score_str}_{iou_str}_{h}x{w}"

                run_RCNN_extractor(dl_key=RCNN_DL_KEY, subm_key="submission_MOT", rcnn_cfg_str=_rcnn_cfg_str)

    send_discord_notification("finished extracting bboxes for MOT")
