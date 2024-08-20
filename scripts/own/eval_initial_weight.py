"""
Script to run an evaluation of the initial weights of the tracks
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

CONFIG_FILE = "./configs/DGS/eval_const_track_weight.yaml"

# 0.0 was done in parameter search
# 1.0 should be pretty much pointless, because every new track is preferred
INITIAL_WEIGHTS: list[float] = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

DL_KEYS: list[str] = [
    "dgs_pt21_rcnn_256x192_val",
    "dgs_Dance_rcnn_256x192_val",
]

KEYS: list[str] = ["iou", "oks", "OSNet"]

IOU_THRESH: float = 0.40  # PT21
SCORE_THRESH: float = 0.85  # PT21
# IOU_THRESH: float = 0.40  # Dance
# SCORE_THRESH: float = 0.75  # Dance


# @torch_memory_analysis
# @MemoryTracker(interval=7.5, top_n=20)
@notify_on_completion_or_error(min_time=30, info="run initial weight")
@t.no_grad()
def run_pt21(config: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
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
@notify_on_completion_or_error(min_time=30)
@t.no_grad()
def run_dance(config: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
    """Set the DanceTrack config."""

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        dataset_path = os.path.normpath(os.path.dirname(os.path.dirname(sub_datapath)))
        dataset_name = os.path.basename(dataset_path)
        pbar_data.set_postfix_str(dataset_name)
        config[dl_key]["data_path"] = sub_datapath

        # make sure to have a unique log dir every time
        orig_log_dir = config["log_dir"]

        # change config data
        config["log_dir"] += f"./{out_key}/{dgs_key}/"
        config["test"]["writer_log_dir_suffix"] = f"./{os.path.basename(sub_datapath)}/"

        # set the new path for the submission file
        subm_key = "submission_MOT"
        config["test"]["submission"] = [subm_key]
        config[subm_key]["file"] = os.path.abspath(
            os.path.normpath(f"{os.path.dirname(dataset_path)}./results_{out_key}_{dgs_key}/{dataset_name}.txt")
        )

        if os.path.exists(config[subm_key]["file"]):
            # reset the original log dir
            config["log_dir"] = orig_log_dir
            continue

        run(config=config, dl_key=dl_key, dgs_key=dgs_key)

        # reset the original log dir
        config["log_dir"] = orig_log_dir


@t.no_grad()
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
    print(f"Cuda available: {t.cuda.is_available()}")

    cfg = load_config(CONFIG_FILE)
    iou_str = f"{int(IOU_THRESH * 100):03d}"
    score_str = f"{int(SCORE_THRESH * 100):03d}"

    for DL_KEY in DL_KEYS:
        print(f"Evaluating on the ground-truth evaluation dataset with config: {DL_KEY}")

        _crop_h, _crop_w = cfg[DL_KEY]["crop_size"]

        # run all of IoU, OKS, and possibly visual similarity
        for DGS_KEY in (pbar_key := tqdm(KEYS, desc="similarities")):
            pbar_key.set_postfix_str(DGS_KEY)

            # run over all initial weights
            for INIT_WEIGHT in (pbar_weights := tqdm(INITIAL_WEIGHTS, desc="initial weights")):
                init_weight_str = f"{int(INIT_WEIGHT * 100):03d}"
                pbar_weight.set_postfix_str(init_weight_str)
                # set name
                cfg["name"] = f"Evaluate-Initial-Track-Weight-{init_weight_str}-{DL_KEY}-{DGS_KEY}"

                # set initial weight
                cfg[DGS_KEY]["new_track_weight"] = INIT_WEIGHT

                if "pt21" in DL_KEY:
                    base_path = os.path.normpath(
                        f"./data/PoseTrack21/posetrack_data/{_crop_h}x{_crop_w}_rcnn_{score_str}_{iou_str}_val/"  # fixme
                    )
                    cfg[DL_KEY]["base_path"] = base_path
                    data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]
                    assert len(data_paths)

                    run_pt21(
                        config=cfg,
                        dl_key=DL_KEY,
                        paths=data_paths,
                        out_key=f"{DL_KEY}_{score_str}_{iou_str}_init_{init_weight_str}",
                        dgs_key=DGS_KEY,
                    )
                elif "Dance" in DL_KEY:
                    rcnn_cfg_str = f"rcnn_{score_str}_{iou_str}_{_crop_h}x{_crop_w}"
                    cfg[DL_KEY]["crop_key"] = rcnn_cfg_str

                    data_paths = [
                        os.path.normpath(os.path.join(p, f"./{rcnn_cfg_str}.txt"))
                        for p in glob(cfg[DL_KEY]["base_path"])
                    ]
                    assert len(data_paths)

                    run_dance(
                        config=cfg,
                        dl_key=DL_KEY,
                        paths=data_paths,
                        out_key=f"{DL_KEY}_{score_str}_{iou_str}_init_{init_weight_str}",
                        dgs_key=DGS_KEY,
                    )
                else:
                    raise NotImplementedError

        send_discord_notification("finished eval initial track weight")
