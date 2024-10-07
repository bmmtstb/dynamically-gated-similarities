"""
Use the DGS module to evaluate the performance of the individual similarities.

Every similarity is evaluated using the DGS module on its own with an alpha value of 1.

The evaluation is run over the ground-truth evaluation set of the PT21 dataset,
and additionally over the PT21 evaluation set but using the RCNN-dataloader to get the predictions.
"""

# pylint: disable=R0801

import os
from glob import glob

import torch as t
from tqdm import tqdm

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers
from dgs.utils.types import Config
from dgs.utils.utils import HidePrint, notify_on_completion_or_error, send_discord_notification

CONFIG_FILE = "./configs/DGS/eval_const_single_similarities.yaml"

# DL_KEY -> list of DGS_KEYS
DL_KEYS: dict[str, list[str]] = {
    "dgs_pt21_gt_256x192_val": [
        "iou",
        "oks",
        "OSNet",
        # "OSNetAIN",
        "Resnet50",
        # "Resnet152",
        # "OSNetAIN_CrossDomainDuke",
        # "OSNetIBN_CrossDomainDuke",
        # "OSNetAIN_CrossDomainMSMT17",
    ],
    "dgs_pt21_gt_256x128_val": [
        "iou",
        "oks",
        "OSNet",
        # "OSNetAIN",
        # "Resnet50",
        # "Resnet152",
        # "OSNetAIN_CrossDomainDuke",
        # "OSNetIBN_CrossDomainDuke",
        # "OSNetAIN_CrossDomainMSMT17",
    ],
    "dgs_Dance_gt_256x192_val": [
        "iou",
        "OSNet",
        # "OSNetAIN",
        # "Resnet50",
        # "Resnet152",
        # "OSNetAIN_CrossDomainDuke",
        # "OSNetIBN_CrossDomainDuke",
        # "OSNetAIN_CrossDomainMSMT17",
    ],
    "dgs_Dance_gt_256x192_train": [
        "iou",
        "OSNet",
        # "OSNetAIN",
        # "Resnet50",
        # "Resnet152",
        # "OSNetAIN_CrossDomainDuke",
        # "OSNetIBN_CrossDomainDuke",
        # "OSNetAIN_CrossDomainMSMT17",
    ],
}

# DL_KEY -> (list of SCORE_THRESHS, list of IOU_THRESHS, list of DGS_KEYS)
RCNN_DL_KEYS: dict[str, tuple[list[float], list[float], list[str]]] = {
    "dgs_pt21_rcnn_256x192_val": (
        [0.85, 0.90, 0.95, 0.99],  # [0.85]
        [0.20, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # [0.4]
        [
            "iou",
            "OSNet",
            # "OSNetAIN",
            # "Resnet50",
            # "Resnet152",
            # "OSNetAIN_CrossDomainDuke",
            # "OSNetIBN_CrossDomainDuke",
            # "OSNetAIN_CrossDomainMSMT17",
        ],
    ),
    "dgs_pt21_rcnn_256x128_val": (
        [0.85],
        [0.4],
        [
            "iou",
            "OSNet",
            # "OSNetAIN",
            # "Resnet50",
            # "Resnet152",
            # "OSNetAIN_CrossDomainDuke",
            # "OSNetIBN_CrossDomainDuke",
            # "OSNetAIN_CrossDomainMSMT17",
        ],
    ),
    "dgs_Dance_rcnn_256x192_val": (
        [0.70, 0.75],
        [0.35],
        [
            "iou",
            "OSNet",
            # "OSNetAIN",
            # "Resnet50",
            # "Resnet152",
            # "OSNetAIN_CrossDomainDuke",
            # "OSNetIBN_CrossDomainDuke",
            # "OSNetAIN_CrossDomainMSMT17",
        ],
    ),
    # "dgs_Dance_rcnn_256x192_test": (
    #     [0.70, 0.75],
    #     [0.35],
    #     [
    #         "iou",
    #         "OSNet",
    #         # "OSNetAIN",
    #         # "Resnet50",
    #         # "Resnet152",
    #     ],
    # ),
}


@notify_on_completion_or_error(min_time=30, info="single")
def single_run_pt21(config: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
    """Set the PT21 config."""
    crop_h, crop_w = config[dl_key]["crop_size"]
    config[dl_key]["crops_folder"] = (
        config[dl_key]["base_path"]
        .replace("posetrack_data", f"crops/{crop_h}x{crop_w}")
        .replace(f"{crop_h}x{crop_w}_", "")  # remove redundant from crop folder name iff existing
    )

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        pbar_data.set_postfix_str(os.path.basename(sub_datapath))

        # change config data
        config[dl_key]["data_path"] = sub_datapath
        log_dir_suffix = f"./{out_key}/{dgs_key}/"
        config["log_dir_suffix"] = log_dir_suffix
        config["test"]["writer_log_dir_suffix"] = f"./{os.path.basename(sub_datapath)}/"

        # set the new path for the out file in the log_dir
        subm_key = "submission_pt21"
        config["test"]["submission"] = [subm_key]
        config[subm_key]["file"] = os.path.abspath(
            os.path.normpath(
                os.path.join(config["log_dir"], log_dir_suffix, f"./results_json/{os.path.basename(sub_datapath)}")
            )
        )

        if os.path.exists(config[subm_key]["file"]):
            continue

        run(config=config, dl_key=dl_key, dgs_key=dgs_key)


@notify_on_completion_or_error(min_time=30, info="single")
def single_run_dance(config: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
    """Set the DanceTrack config."""

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        dataset_path = os.path.normpath(os.path.dirname(os.path.dirname(sub_datapath)))
        dataset_name = os.path.basename(dataset_path)
        pbar_data.set_postfix_str(dataset_name)
        config[dl_key]["data_path"] = sub_datapath

        # change config data
        config["log_dir_suffix"] = f"./{out_key}/{dgs_key}/"
        config["test"]["writer_log_dir_suffix"] = f"./{dataset_name}/"

        # set the new path for the submission file
        subm_key = "submission_MOT"
        config["test"]["submission"] = [subm_key]
        config[subm_key]["file"] = os.path.abspath(
            os.path.normpath(f"{os.path.dirname(dataset_path)}/results_{out_key}_{dgs_key}/{dataset_name}.txt")
        )

        if os.path.exists(config[subm_key]["file"]):
            continue

        run(config=config, dl_key=dl_key, dgs_key=dgs_key)


def run(config: Config, dl_key: str, dgs_key: str) -> None:
    """Main function to run the code after all the parameters are set."""
    config[dl_key]["load_image_crops"] = dgs_key not in ["iou", "oks"]

    with HidePrint():
        # validation dataset
        val_dl = module_loader(config=config, module_type="dataloader", key=dl_key)

        # will load all the similarity modules
        model: DGSModule = module_loader(config=config, module_type="dgs", key=dgs_key)
        close_all_layers(model)

        engine = DGSEngine(config=config, path=["engine"], model=model, test_loader=val_dl)

    engine.test()

    # end processes
    engine.terminate()


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")
    cfg = load_config(CONFIG_FILE)

    # ## #
    # GT #
    # ## #

    print("Evaluating single models on the GT evaluation datasets")
    for DL_KEY, DGS_KEYS in (pbar_dl := tqdm(DL_KEYS.items(), desc="Dataloader", leave=False)):
        pbar_dl.set_postfix_str(DL_KEY)

        # IoU, OKS, and visual similarity
        for DGS_KEY in (pbar_key := tqdm(DGS_KEYS, desc="similarities", leave=False)):
            pbar_key.set_postfix_str(DGS_KEY)

            if DGS_KEY == "oks":
                # not possible for gt data
                continue

            if "pt21" in DL_KEY:
                data_paths = [f.path for f in os.scandir(cfg[DL_KEY]["base_path"]) if f.is_file()]
                assert len(data_paths)
                single_run_pt21(config=cfg, dl_key=DL_KEY, paths=data_paths, out_key=DL_KEY, dgs_key=DGS_KEY)
            elif "Dance" in DL_KEY:
                if DGS_KEY == "oks":
                    continue
                data_paths = [os.path.normpath(p) for p in glob(cfg[DL_KEY]["base_path"])]
                assert len(data_paths)
                single_run_dance(config=cfg, dl_key=DL_KEY, paths=data_paths, out_key=DL_KEY, dgs_key=DGS_KEY)
            else:
                raise NotImplementedError

    # #### #
    # RCNN #
    # #### #

    print("Evaluating single models on the validation-data using KeypointRCNN as prediction backbone")
    for RCNN_DL_KEY, (SCORE_THRESHS, IOU_THRESHS, DGS_KEYS) in (
        pbar_dl := tqdm(RCNN_DL_KEYS.items(), desc="RCNN Dataloaders", leave=False)
    ):
        pbar_dl.set_postfix_str(f"{RCNN_DL_KEY}")

        for score_thresh in (pbar_score_thresh := tqdm(SCORE_THRESHS, desc="Score Thresh", leave=False)):
            score_s = f"{int(score_thresh * 100):03d}"
            pbar_score_thresh.set_postfix_str(os.path.basename(score_s))

            for iou_thresh in (pbar_iou_thresh := tqdm(IOU_THRESHS, desc="IoU Thresh", leave=False)):
                iou_s = f"{int(iou_thresh * 100):03d}"
                pbar_iou_thresh.set_postfix_str(os.path.basename(iou_s))

                # IoU, OKS, and visual similarity
                for DGS_KEY in (pbar_key := tqdm(DGS_KEYS, desc="similarities", leave=False)):
                    pbar_key.set_postfix_str(DGS_KEY)

                    cfg = load_config(CONFIG_FILE)
                    cfg["name"] = f"Evaluate-Single-{DGS_KEY}"
                    _crop_h, _crop_w = cfg[RCNN_DL_KEY]["crop_size"]
                    if "pt21" in RCNN_DL_KEY:
                        base_path = os.path.normpath(
                            f"./data/PoseTrack21/posetrack_data/{_crop_h}x{_crop_w}_rcnn_{score_s}_{iou_s}_val/"
                        )
                        cfg[RCNN_DL_KEY]["base_path"] = base_path
                        data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]
                        assert len(data_paths), f"There are no paths. base_path: {base_path}"

                        single_run_pt21(
                            config=cfg,
                            dl_key=RCNN_DL_KEY,
                            paths=data_paths,
                            out_key=f"{RCNN_DL_KEY}_{score_s}_{iou_s}",
                            dgs_key=DGS_KEY,
                        )
                    elif "Dance" in RCNN_DL_KEY:
                        rcnn_cfg_str = f"rcnn_{score_s}_{iou_s}_{_crop_h}x{_crop_w}"
                        cfg[RCNN_DL_KEY]["crop_key"] = rcnn_cfg_str

                        data_paths = [
                            os.path.normpath(os.path.join(p, f"./{rcnn_cfg_str}.txt"))
                            for p in glob(cfg[RCNN_DL_KEY]["base_path"])
                        ]
                        assert len(data_paths), f"There are no paths. rcnn_cfg_str: {rcnn_cfg_str}"

                        single_run_dance(
                            config=cfg,
                            dl_key=RCNN_DL_KEY,
                            paths=data_paths,
                            out_key=f"{RCNN_DL_KEY}_{score_s}_{iou_s}",
                            dgs_key=DGS_KEY,
                        )
                    else:
                        raise NotImplementedError

    send_discord_notification("finished eval single")
