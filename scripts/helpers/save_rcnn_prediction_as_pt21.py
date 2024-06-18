"""Run the RCNN backbone over the whole dataset and save the results, so they don't have to be recomputed every time."""

import os.path
from glob import glob

import torch
from torchvision.io import write_jpeg
from torchvision.transforms.v2.functional import convert_image_dtype
from tqdm import tqdm

from dgs.models.loader import module_loader
from dgs.models.submission import PoseTrack21Submission
from dgs.utils.config import DEF_VAL, load_config
from dgs.utils.files import mkdir_if_missing, read_json
from dgs.utils.state import State
from dgs.utils.types import Config

CONFIG_FILE: str = "./configs/helpers/predict_rcnn.yaml"
DL_KEY: str = "RCNN_Backbone"

# IN images: "./data/PoseTrack21/crops/hxw/val/DATASET/*.jpg"
# OUT predictions: "./data/PoseTrack21/posetrack_data/rcnn_prediction_XXX/DATASET.json"
# OUT cropped image and loc_kp: "./data/PoseTrack21/crops/hxw/rcnn_prediction_XXX/DATASET/*.jpg" and "...*.pt"

if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}")

    config: Config = load_config(CONFIG_FILE)

    h, w = config[DL_KEY]["crop_size"]

    for score_threshold in (pbar_score_thresh := tqdm([0.85, 0.90, 0.95, 0.99], desc="Score-Threshold")):
        pbar_score_thresh.set_postfix_str(str(score_threshold))

        score_str = f"{int(score_threshold * 100):03d}"
        config[DL_KEY]["score_threshold"] = score_threshold

        for iou_threshold in (pbar_iou_thresh := tqdm([0.3, 0.5, 0.7, 0.9], desc="IoU-Threshold")):
            pbar_iou_thresh.set_postfix_str(str(iou_threshold))

            iou_str = f"{int(iou_threshold * 100):03d}"
            config[DL_KEY]["iou_threshold"] = iou_threshold

            datasets: list = glob("./data/PoseTrack21/images/val/**/")

            for dataset_path in (pbar_dataset := tqdm(datasets, desc="datasets", leave=False)):
                ds_name = os.path.basename(os.path.realpath(dataset_path))
                pbar_dataset.set_postfix_str(ds_name)

                # modify the configuration
                config[DL_KEY]["data_path"] = dataset_path
                config["submission"][
                    "file"
                ] = f"./data/PoseTrack21/posetrack_data/rcnn_prediction_{score_str}_{iou_str}/{ds_name}.json"

                if os.path.exists(config["submission"]["file"]):
                    continue

                dataloader = module_loader(config=config, module_class="dataloader", key=DL_KEY)

                submission: PoseTrack21Submission = module_loader(
                    config=config, module_class="submission", key="submission"
                )

                # GT data
                gt_data = read_json(f"./data/PoseTrack21/posetrack_data/val/{ds_name}.json")
                gt_img_id_map = [img["image_id"] for img in gt_data["images"]]  # zero-indexed!
                gt_n_frames: int = gt_data["images"][0]["nframes"]
                gt_vid_id: int = gt_data["images"][0]["vid_id"]

                # create img output folder
                mkdir_if_missing(
                    os.path.dirname(
                        f"./data/PoseTrack21/crops/{h}x{w}/rcnn_prediction_{score_str}_{iou_str}/{ds_name}/"
                    )
                )

                batch: list[State]
                for batch in tqdm(dataloader, desc="batch", leave=False):

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
                        _ = submission.get_image_data(s)

                        assert all(s.filepath[0] == s.filepath[i] for i in range(len(s.filepath)))
                        fp = "/".join(s.filepath[0].split("/")[-4:])

                        B = s.B

                        own_iid = int(s["image_id"][0].item())  # one-indexed

                        gt_img_id: int = gt_img_id_map[own_iid - 1]

                        ignore_regions_x: list[list, list] = gt_data["images"][own_iid - 1]["ignore_regions_x"]
                        ignore_regions_y: list[list, list] = gt_data["images"][own_iid - 1]["ignore_regions_y"]

                        # create dict with all the image data
                        image = {
                            "is_labeled": True,
                            "nframes": gt_n_frames,
                            "image_id": gt_img_id,
                            "id": gt_img_id,
                            "vid_id": gt_vid_id,
                            "file_name": fp,
                            "has_labeled_person": B != 0,
                            "ignore_regions_x": ignore_regions_x,
                            "ignore_regions_y": ignore_regions_y,
                        }

                        submission.data["images"].append(image)

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

                        annos = submission.get_anno_data(s)
                        for a_i, anno in enumerate(annos):
                            annos[a_i]["bbox"] = anno["bboxes"]
                            annos[a_i]["bbox_head"] = anno["bboxes"]
                            annos[a_i]["image_id"] = gt_img_id
                            annos[a_i]["id"] = gt_img_id
                            annos[a_i]["category_id"] = 1
                            # remove unnecessary keys
                            annos[a_i].pop("bboxes", None)
                            annos[a_i].pop("kps", None)
                            annos[a_i].pop("score", None)
                            annos[a_i].pop("scores", None)

                        submission.data["annotations"] += annos
                        # save the image-crops and local key points
                        for i in range(s.B):
                            img_path = (
                                f"./data/PoseTrack21/crops/{h}x{w}/rcnn_prediction_{score_str}/"
                                f"{ds_name}/{gt_img_id}_{s['person_id'][i]}.jpg"
                            )
                            if os.path.exists(img_path):
                                continue
                            write_jpeg(
                                input=convert_image_dtype(s.image_crop[i], torch.uint8).cpu(),
                                filename=img_path,
                                quality=DEF_VAL["images"]["jpeg_quality"],
                            )
                            torch.save(s.keypoints_local[i].unsqueeze(0).cpu(), str(img_path).replace(".jpg", ".pt"))

                submission.save()
