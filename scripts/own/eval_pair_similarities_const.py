"""
Use the DGS module to evaluate the performance of a combination of two similarities.

Every pair of similarities is evaluated using the DGS module on its own with constant alpha values.
Every combination of alpha values in 10% increments is used.

1. [10, 90]
2. [20, 80]
...
9. [90, 10]

The evaluation is run over the ground-truth evaluation set of the PT21 dataset,
and additionally over the PT21 evaluation set but using the RCNN-dataloader to obtain the predictions.
"""

# pylint: disable=R0801

import os

import torch as t
from tqdm import tqdm

from dgs.models.dgs import DGSModule
from dgs.models.engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers
from dgs.utils.types import Config
from dgs.utils.utils import HidePrint, notify_on_completion_or_error, send_discord_notification

CONFIG_FILE = "./configs/DGS/eval_const_pairwise_similarities.yaml"


DGS_KEYS = [
    "iou_oks",
    "iou_OSNet",
    "oks_OSNet",
    "iou_OSNetAIN",
    "oks_OSNetAIN",
    "iou_Resnet50",
    "oks_Resnet50",
    "iou_Resnet152",
    "oks_Resnet152",
]

IOU_THRESH: float = 0.40  # PT21
SCORE_THRESH: float = 0.85  # PT21
INITIAL_WEIGHT: float = 0.00  # FIXME

DL_KEY = "dgs_pt21_gt_256x192_val"
RCNN_DL_KEY = "dgs_pt21_rcnn_256x192_val"


@notify_on_completion_or_error(min_time=30, info="double")
@t.no_grad()
def run(config: Config, dl_key: str, paths: list[str], out_key: str) -> None:
    """Main function to run the code."""
    # Combinations of IoU, OKS, and visual similarity
    for dgs_key in (pbar_key := tqdm(DGS_KEYS, desc="combinations")):
        pbar_key.set_postfix_str(dgs_key)

        config["name"] = f"Evaluate-Pairwise-Combinations-{dgs_key}"

        # set initial weight
        config[dgs_key]["new_track_weight"] = INITIAL_WEIGHT

        # get sub folders or files and analyse them one-by-one
        for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
            pbar_data.set_postfix_str(os.path.basename(sub_datapath))

            config[dl_key]["data_path"] = sub_datapath

            for alpha_1 in (pbar_alpha := tqdm(range(10, 91, 10), desc="alpha", leave=False)):
                pbar_alpha.set_postfix_str(f"[{alpha_1}, {100 - alpha_1}]")

                # set alpha values
                config["combine_sim"]["alpha"] = [alpha_1 / 100.0, (100.0 - alpha_1) / 100.0]

                # make sure to have a unique log dir every time
                orig_log_dir = config["log_dir"]

                # change config data
                config["log_dir"] += f"./{out_key}/{dgs_key}_{alpha_1}_{100-alpha_1}/"
                config["test"]["writer_log_dir_suffix"] = f"./{os.path.basename(sub_datapath)}/"

                # set the new path for the out file in the log_dir
                config["submission"]["file"] = os.path.abspath(
                    os.path.normpath(
                        f"{config['log_dir']}/results_json/{sub_datapath.split('/')[-1].removesuffix('.json')}.json"
                    )
                )

                if os.path.exists(config["submission"]["file"]):
                    # reset the original log dir
                    config["log_dir"] = orig_log_dir
                    continue

                with HidePrint():
                    # validation dataset
                    val_dl = module_loader(config=config, module_class="dataloader", key=dl_key)

                    # will load all the similarity modules
                    model: DGSModule = module_loader(config=config, module_class="dgs", key=dgs_key).cuda()
                    close_all_layers(model)

                    engine = DGSEngine(config=config, model=model, test_loader=val_dl)

                engine.test()

                # end processes
                engine.terminate()
                del val_dl, model, engine

                # reset the original log dir
                config["log_dir"] = orig_log_dir
        send_discord_notification(f"eval double completed combination: {dgs_key}")


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")

    print("Evaluating pairwise models on the PT21 ground-truth evaluation dataset")

    cfg = load_config(CONFIG_FILE)
    base_path = cfg[DL_KEY]["base_path"]
    data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]
    run(config=cfg, dl_key=DL_KEY, paths=data_paths, out_key=DL_KEY)

    print("Evaluating pairwise models on the PT21 validation-dataset using KeypointRCNN as prediction backbone")
    score_str = f"{int(SCORE_THRESH * 100):03d}"
    iou_str = f"{int(IOU_THRESH * 100):03d}"

    rcnn_cfg_str = f"rcnn_{score_str}_{iou_str}_val"

    cfg = load_config(CONFIG_FILE)
    crop_h, crop_w = cfg[RCNN_DL_KEY]["crop_size"]

    base_path = f"./data/PoseTrack21/posetrack_data/{crop_h}x{crop_w}_{rcnn_cfg_str}/"
    if not os.path.isdir(base_path):
        send_discord_notification("Double - base path not found")
        raise ValueError("Double - base path not found")
    cfg[RCNN_DL_KEY]["base_path"] = base_path
    cfg[RCNN_DL_KEY]["crops_folder"] = f"./data/PoseTrack21/crops/{crop_h}x{crop_w}/{rcnn_cfg_str}/"
    data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]
    assert len(data_paths) > 0, f"No files found in the base_path: {base_path}"
    run(config=cfg, dl_key=DL_KEY, paths=data_paths, out_key=f"dgs_pt21_{rcnn_cfg_str}")

    send_discord_notification("finished eval double")
