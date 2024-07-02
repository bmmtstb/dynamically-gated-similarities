"""
Use the DGS module to evaluate the performance of the individual similarities.

Every similarity is evaluated using the DGS module on its own with an alpha value of 1.

The evaluation is run over the ground-truth evaluation set of the PT21 dataset,
and additionally over the PT21 evaluation set but using the RCNN-dataloader to obtain the predictions.
"""

# pylint: disable=R0801

import os
from glob import glob

import torch
from tqdm import tqdm

from dgs.models.dgs import DGSModule
from dgs.models.engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers
from dgs.utils.types import Config
from dgs.utils.utils import HidePrint

CONFIG_FILE = "./configs/DGS/eval_const_single_similarities.yaml"

DL_KEYS: list[str] = [
    "dgs_pt21_gt_256x192",
    "dgs_pt21_gt_256x128",
    "dgs_Dance_gt",
]
RCNN_DL_KEYS: list[str] = [
    "dgs_pt21_rcnn_256x192",
    "dgs_pt21_rcnn_256x128",
    "dgs_Dance_rcnn_256x192",
]

KEYS: list[str] = [
    "iou",
    "oks",
    "OSNet",
    "OSNetAIN",
    "Resnet50",
    "Resnet152",
    "OSNetAIN_CrossDomainDuke",
    "OSNetIBN_CrossDomainDuke",
    "OSNetAIN_CrossDomainMSMT17",
]
SCORE_THRESHS: list[float] = [0.85, 0.90, 0.95]
# IOU_THRESHS: list[float] = [0.5, 0.6, 0.7, 0.8]
IOU_THRESHS: list[float] = [1.0]


# @torch_memory_analysis
# @MemoryTracker(interval=7.5, top_n=20)
@torch.no_grad()
def run_pt21(config: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
    """Set the PT21 config."""
    crop_h, crop_w = config[dl_key]["crop_size"]
    config[dl_key]["crops_folder"] = config[dl_key]["base_path"].replace("posetrack_data", f"crops/{crop_h}x{crop_w}")

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        pbar_data.set_postfix_str(os.path.basename(sub_datapath))
        # make sure to have a unique log dir every time
        orig_log_dir = config["log_dir"]

        # change config data
        config[dl_key]["data_path"] = sub_datapath
        config["log_dir"] += f"./{out_key}/{dgs_key}/"
        config["test"]["submission"] = ["submission_pt21"]

        # set the new path for the out file in the log_dir
        subm_key = "submission_pt21"
        config[subm_key]["file"] = os.path.abspath(
            os.path.normpath(
                f"{config['log_dir']}/results_json/{sub_datapath.split('/')[-1].removesuffix('.json')}.json"
            )
        )

        if os.path.exists(config[subm_key]["file"]):
            # reset the original log dir
            config["log_dir"] = orig_log_dir
            continue

        run(config=config, dl_key=dl_key, dgs_key=dgs_key)

        # reset the original log dir
        config["log_dir"] = orig_log_dir


# @torch_memory_analysis
@torch.no_grad()
def run_dance(config: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
    """Set the DanceTrack config."""

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(sub_datapath)))
        pbar_data.set_postfix_str(dataset_name)
        config[dl_key]["data_path"] = sub_datapath

        # make sure to have a unique log dir every time
        orig_log_dir = config["log_dir"]

        # change config data
        config["log_dir"] += f"./{out_key}/{dgs_key}/"
        config["test"]["submission"] = ["submission_MOT"]

        # set the new path for the submission file - uses PT21 submission !
        subm_key = "submission_MOT"
        config[subm_key]["file"] = os.path.abspath(
            os.path.normpath(f"{config['log_dir']}/results_txt/{dataset_name}.txt")
        )

        if os.path.exists(config[subm_key]["file"]):
            # reset the original log dir
            config["log_dir"] = orig_log_dir
            continue

        run(config=config, dl_key=dl_key, dgs_key=dgs_key)

        # reset the original log dir
        config["log_dir"] = orig_log_dir


@torch.no_grad()
def run(config: Config, dl_key: str, dgs_key: str) -> None:
    """Main function to run the code after all the parameters are set."""
    with HidePrint():
        # validation dataset
        val_dl = module_loader(config=config, module_class="dataloader", key=dl_key)

        # will load all the similarity modules
        model: DGSModule = module_loader(config=config, module_class="dgs", key=dgs_key)
        close_all_layers(model)

        engine = DGSEngine(config=config, model=model, test_loader=val_dl)

    engine.test()

    # end processes
    engine.terminate()


if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}")

    cfg = load_config(CONFIG_FILE)
    for DL_KEY in DL_KEYS:
        print(f"Evaluating on the PT21 ground-truth evaluation dataset with config: {DL_KEY}")
        # IoU, OKS, and visual similarity
        for DGS_KEY in (pbar_key := tqdm(KEYS, desc="similarities")):
            pbar_key.set_postfix_str(DGS_KEY)

            if "pt21" in DL_KEY:
                data_paths = [f.path for f in os.scandir(cfg[DL_KEY]["base_path"]) if f.is_file()]
                assert len(data_paths)
                run_pt21(config=cfg, dl_key=DL_KEY, paths=data_paths, out_key=DL_KEY, dgs_key=DGS_KEY)
            elif "Dance" in DL_KEY:
                if DGS_KEY == "oks":
                    continue
                data_paths = [os.path.normpath(p) for p in glob(cfg[DL_KEY]["base_path"])]
                assert len(data_paths)
                run_dance(config=cfg, dl_key=DL_KEY, paths=data_paths, out_key=DL_KEY, dgs_key=DGS_KEY)
            else:
                raise NotImplementedError

    for RCNN_DL_KEY in RCNN_DL_KEYS:
        print(f"Evaluating on the PT21 val-dataset using KeypointRCNN with config: {RCNN_DL_KEY}")
        for score_thresh in (pbar_score_thresh := tqdm(SCORE_THRESHS, desc="Score Thresh")):
            score_str = f"{int(score_thresh * 100):03d}"
            pbar_score_thresh.set_postfix_str(os.path.basename(score_str))
            for iou_thresh in (pbar_iou_thresh := tqdm(IOU_THRESHS, desc="IoU Thresh")):
                iou_str = f"{int(iou_thresh * 100):03d}"
                pbar_iou_thresh.set_postfix_str(os.path.basename(iou_str))
                # IoU, OKS, and visual similarity
                for DGS_KEY in (pbar_key := tqdm(KEYS, desc="similarities")):
                    pbar_key.set_postfix_str(DGS_KEY)

                    cfg = load_config(CONFIG_FILE)
                    cfg["name"] = f"Evaluate-Single-{DGS_KEY}"
                    if "pt21" in RCNN_DL_KEY:
                        base_path = os.path.normpath(
                            f"./data/PoseTrack21/posetrack_data/rcnn_{score_str}_{iou_str}_val/"
                        )
                        cfg[RCNN_DL_KEY]["base_path"] = base_path
                        data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]

                        run_pt21(
                            config=cfg,
                            dl_key=RCNN_DL_KEY,
                            paths=data_paths,
                            out_key=f"{RCNN_DL_KEY}_{score_str}_{iou_str}_val",
                            dgs_key=DGS_KEY,
                        )
                    elif "Dance" in RCNN_DL_KEY:
                        _crop_h, _crop_w = cfg[RCNN_DL_KEY]["crop_size"]
                        rcnn_cfg_str = f"rcnn_{score_str}_{iou_str}_{_crop_h}x{_crop_w}"
                        cfg[RCNN_DL_KEY]["crop_key"] = rcnn_cfg_str

                        data_paths = [
                            os.path.join(p, f"./{rcnn_cfg_str}.txt") for p in glob(cfg[RCNN_DL_KEY]["base_path"])
                        ]

                        run_dance(
                            config=cfg,
                            dl_key=RCNN_DL_KEY,
                            paths=data_paths,
                            out_key=f"{RCNN_DL_KEY}_{score_str}_{iou_str}_val",
                            dgs_key=DGS_KEY,
                        )
                    else:
                        raise NotImplementedError
