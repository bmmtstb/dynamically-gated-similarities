"""
Use the DGS module to evaluate the performance of a combination of two similarities.

Every pair of similarities is evaluated using the DGS module on its own with constant alpha values.
Every combination of alpha values in 10% increments is used.

1. [10, 10, 80]
2. [10, 20, 70]
...
35. [80, 10, 10]

The evaluation is run over the ground-truth evaluation set of the PT21 dataset,
and additionally over the PT21 evaluation set but using the RCNN-dataloader to obtain the predictions.
"""

# pylint: disable=R0801

import os
from glob import glob

import torch as t
from tqdm import tqdm

from dgs.models.dgs import DGSModule
from dgs.models.engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers
from dgs.utils.types import Config
from dgs.utils.utils import HidePrint, notify_on_completion_or_error, send_discord_notification

CONFIG_FILE = "./configs/DGS/eval_const_triplet_similarities.yaml"

GT_DL_KEYS: dict[str, list[str]] = {
    "dgs_pt21_gt_256x192_val": [
        "iou_oks_OSNet",
        #        "iou_oks_OSNetAIN",
        #        "iou_oks_Resnet50",
        #        "iou_oks_Resnet152",
    ],
    # within DanceTrack there is no GT data for the OKS available, therefore it is skipped
}

RCNN_DL_KEYS: list[tuple[str, float, float, float, list[str]]] = [
    (
        "dgs_pt21_rcnn_256x192_val",
        0.85,
        0.40,
        0.00,
        [
            "iou_oks_OSNet",
            #            "iou_oks_OSNetAIN",
            #            "iou_oks_Resnet50",
            #            "iou_oks_Resnet152",
        ],
    ),
    (
        "dgs_Dance_rcnn_256x192_val",
        0.75,
        0.35,
        0.00,
        [
            "iou_oks_OSNet",
            #           "iou_oks_OSNetAIN",
            #           "iou_oks_Resnet50",
            #           "iou_oks_Resnet152",
        ],
    ),
]


def get_combinations() -> list[list[int]]:
    """Get all triplet combinations as list of integers in range 10-90, all summing to 100."""
    combinations = []
    for a1 in range(10, 91, 10):
        for a2 in range(10, 91, 10):
            a3 = 100 - a1 - a2
            if a3 <= 0 or a3 >= 100:
                continue
            combinations.append([a1, a2, a3])
    return combinations


@notify_on_completion_or_error(min_time=30, info="triplet")
def run_pt21(config: Config, dl_key: str, paths: list, dgs_key: str, out_key: str, initial_weights: float) -> None:
    """Main function to run the code."""
    config["name"] = f"Evaluate-Triplet-Combinations-{dl_key}-{dgs_key}"
    # set initial weight
    config[dgs_key]["new_track_weight"] = initial_weights

    # get sub folders or files and analyse them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        pbar_data.set_postfix_str(os.path.basename(sub_datapath))

        config[dl_key]["data_path"] = sub_datapath

        for combination in (pbar_alpha := tqdm(get_combinations(), desc="alphas", leave=False)):
            pbar_alpha.set_postfix_str(f"{combination}")

            # set alpha values
            config["combine_sim"]["alpha"] = [c / 100.0 for c in combination]

            # change config data
            log_dir_suffix = f"./{out_key}/{dgs_key}_{'_'.join(str(c) for c in combination)}/"
            config["log_dir_suffix"] = log_dir_suffix
            config["test"]["writer_log_dir_suffix"] = f"./{os.path.basename(sub_datapath)}/"

            # set the new path for the out file in the log_dir
            config["submission"]["file"] = os.path.abspath(
                os.path.normpath(
                    os.path.join(config["log_dir"], log_dir_suffix, f"./results_json/{os.path.basename(sub_datapath)}")
                )
            )

            if os.path.exists(config["submission"]["file"]):
                continue

            run(config=config, dl_key=dl_key, dgs_key=dgs_key)


@notify_on_completion_or_error(min_time=30, info="single")
def run_dance(config: Config, dl_key: str, paths: list, dgs_key: str, out_key: str, initial_weights: float) -> None:
    """Set the DanceTrack config."""
    config["name"] = f"Evaluate-Pairwise-Combinations-{dl_key}-{dgs_key}"
    # set initial weight
    config[dgs_key]["new_track_weight"] = initial_weights

    # get sub folders or files and analyse them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        pbar_data.set_postfix_str(os.path.basename(sub_datapath))

        dataset_path = os.path.normpath(os.path.dirname(os.path.dirname(sub_datapath)))
        dataset_name = os.path.basename(dataset_path)

        config[dl_key]["data_path"] = sub_datapath

        for combination in (pbar_alpha := tqdm(get_combinations(), desc="alphas", leave=False)):
            pbar_alpha.set_postfix_str(f"{combination}")

            # set alpha values
            config["combine_sim"]["alpha"] = [c / 100.0 for c in combination]

            # change config data
            log_dir_suffix = f"./results_{out_key}/{dgs_key}_{'_'.join(str(c) for c in combination)}/"
            config["log_dir_suffix"] = log_dir_suffix
            config["test"]["writer_log_dir_suffix"] = f"./{os.path.basename(sub_datapath)}/"

            # set the new path for the out file in the log_dir
            config["submission"]["module_name"] = "MOT"
            config["submission"]["file"] = os.path.abspath(
                os.path.normpath(os.path.join(config[dl_key]["dataset_path"], log_dir_suffix, f"./{dataset_name}.txt"))
            )

            if os.path.exists(config["submission"]["file"]):
                # reset the original log dir
                continue

            run(config=config, dl_key=dl_key, dgs_key=dgs_key)


def run(config: Config, dl_key: str, dgs_key: str) -> None:
    """Main function to run the code."""

    with HidePrint():
        # validation dataset
        val_dl = module_loader(config=config, module_type="dataloader", key=dl_key)

        # will load all the similarity modules
        model: DGSModule = module_loader(config=config, module_type="dgs", key=dgs_key).cuda()
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

    print("Evaluating triplet models on the GT evaluation-data")
    for DL_KEY, DGS_KEYS in (pbar_dl := tqdm(GT_DL_KEYS.items(), desc="GT Dataloaders", leave=False)):
        pbar_dl.set_postfix_str(DL_KEY)

        for DGS_KEY in (pbar_key := tqdm(DGS_KEYS, desc="DGS Keys", leave=False)):
            pbar_key.set_postfix_str(DGS_KEY)

            if "pt21" in DL_KEY:
                data_paths = [f.path for f in os.scandir(cfg[DL_KEY]["paths"]) if f.is_file()]
                assert len(data_paths)
                run_pt21(
                    config=cfg, dl_key=DL_KEY, paths=data_paths, dgs_key=DGS_KEY, out_key=DL_KEY, initial_weights=0.0
                )
            elif "Dance" in DL_KEY:
                if DGS_KEY == "oks":
                    continue
                data_paths = [os.path.normpath(p) for p in glob(cfg[DL_KEY]["paths"])]
                assert len(data_paths)
                run_dance(
                    config=cfg, dl_key=DL_KEY, paths=data_paths, dgs_key=DGS_KEY, out_key=DL_KEY, initial_weights=0.0
                )
            else:
                raise NotImplementedError

    # #### #
    # RCNN #
    # #### #

    print("Evaluating triplet models on the validation-data using KeypointRCNN as prediction backbone")
    for RCNN_DL_KEY, SCORE_THRESH, IOU_THRESH, INIT_WEIGHT, DGS_KEYS in (
        pbar_dl := tqdm(RCNN_DL_KEYS, desc="RCNN Dataloaders", leave=False)
    ):
        pbar_dl.set_postfix_str(f"{RCNN_DL_KEY}_S{SCORE_THRESH}_I{IOU_THRESH}_W{INIT_WEIGHT}")

        cfg = load_config(CONFIG_FILE)
        crop_h, crop_w = cfg[RCNN_DL_KEY]["crop_size"]
        score_s: str = f"{int(SCORE_THRESH * 100):03d}"
        iou_s: str = f"{int(IOU_THRESH * 100):03d}"

        for DGS_KEY in (pbar_key := tqdm(DGS_KEYS, desc="DGS Keys", leave=False)):
            pbar_key.set_postfix_str(DGS_KEY)

            if "pt21" in RCNN_DL_KEY:
                base_paths = os.path.join(
                    cfg[RCNN_DL_KEY]["dataset_path"],
                    f"./posetrack_data/{crop_h}x{crop_w}_rcnn_{score_s}_{iou_s}_val/",
                )
                if not os.path.isdir(base_paths):
                    send_discord_notification("Double - base path not found")
                    raise ValueError("Double - base path not found")

                cfg[RCNN_DL_KEY]["paths"] = base_paths
                cfg[RCNN_DL_KEY]["crops_folder"] = os.path.join(
                    cfg[RCNN_DL_KEY]["dataset_path"], f"./crops/{crop_h}x{crop_w}/rcnn_{score_s}_{iou_s}_val/"
                )

                data_paths = [f.path for f in os.scandir(base_paths) if f.is_file()]
                assert len(data_paths) > 0, f"No files found in the paths: {base_paths}"
                run_pt21(
                    config=cfg,
                    dl_key=RCNN_DL_KEY,
                    paths=data_paths,
                    dgs_key=DGS_KEY,
                    out_key=f"{RCNN_DL_KEY}_{score_s}_{iou_s}",
                    initial_weights=INIT_WEIGHT,
                )

            elif "Dance" in RCNN_DL_KEY:
                base_paths = os.path.join(
                    cfg[RCNN_DL_KEY]["dataset_path"], f"./dancetrack*/det/rcnn_{score_s}_{iou_s}_{crop_h}x{crop_w}.txt"
                )
                cfg[RCNN_DL_KEY]["paths"] = base_paths
                cfg[RCNN_DL_KEY]["crop_key"] = f"rcnn_{score_s}_{iou_s}_{crop_h}x{crop_w}"

                data_paths = [os.path.normpath(p) for p in glob(base_paths)]
                assert len(data_paths) > 0, f"No files found in the paths: {base_paths}"
                run_dance(
                    config=cfg,
                    dl_key=RCNN_DL_KEY,
                    paths=data_paths,
                    dgs_key=DGS_KEY,
                    out_key=f"{RCNN_DL_KEY}_{score_s}_{iou_s}",
                    initial_weights=INIT_WEIGHT,
                )

            else:
                raise NotImplementedError

    send_discord_notification("finished eval triple")
