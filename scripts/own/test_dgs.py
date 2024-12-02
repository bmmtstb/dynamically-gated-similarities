"""
Given the configuration, run the DGS module on test data to predict the results.
"""

import os
from glob import glob

import torch as t
from tqdm import tqdm

from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers
from dgs.utils.types import Config
from dgs.utils.utils import HidePrint

CONFIG_FILE: str = "./configs/DGS/test_dgs.yaml"

# name => dl_key, submission, combine_flag, alpha, sim_mods
SETTINGS: dict[str, tuple[str, str, str, list, list[str]]] = {
    "dance_static_iou": (
        "test_dl_Dance_256x192_rcnn_075_035",
        "submission_MOT",
        "static",
        [1.0],
        ["box_xywh_sim"],
    ),
    "dance_static_oks": (
        "test_dl_Dance_256x192_rcnn_075_035",
        "submission_MOT",
        "static",
        [1.0],
        ["pose_sim_coco"],
    ),
    "dance_static_020_060_020": (
        "test_dl_Dance_256x192_rcnn_075_035",
        "submission_MOT",
        "static",
        [0.2, 0.6, 0.2],
        ["box_xywh_sim", "pose_sim_coco", "OSNet_sim"],
    ),
    "dance_dynamic_iou_oks": (
        "test_dl_Dance_256x192_rcnn_075_035",
        "submission_MOT",
        "dynamic",
        [("Dance", "box_fc1_Sigmoid", 4), ("pt21", "pose_coco_fc1_Sigmoid", 4)],
        ["box_xywh_sim", "pose_sim_coco"],
    ),
    "dance_dynamic_iou_oks_OSNet": (
        "test_dl_Dance_256x192_rcnn_075_035",
        "submission_MOT",
        "dynamic",
        [
            ("Dance", "box_fc2_2Sigmoid", 2),
            ("pt21", "pose_coco_fc2_2Sigmoid", 4),
            ("Dance", "visual_osn_fc5_4ReLUSigmoid", 4),
        ],
        ["box_xywh_sim", "pose_sim_coco", "OSNet_sim"],
    ),
    "pt21_static_oks": (
        "test_dl_pt21_256x192_rcnn_085_040",
        "submission_pt21",
        "static",
        [1.0],
        ["pose_sim_coco"],
    ),
    "pt21_static_oks_OSNet_070_030": (
        "test_dl_pt21_256x192_rcnn_085_040",
        "submission_pt21",
        "static",
        [0.7, 0.3],
        ["pose_sim_coco", "OSNet_sim"],
    ),
    "pt21_static_010_070_020": (
        "test_dl_pt21_256x192_rcnn_085_040",
        "submission_pt21",
        "static",
        [0.1, 0.7, 0.2],
        ["box_xywh_sim", "pose_sim_coco", "OSNet_sim"],
    ),
    "pt21_dynamic_iou_OSNet": (
        "test_dl_pt21_256x192_rcnn_085_040",
        "submission_pt21",
        "dynamic",
        [("Dance", "box_fc1_Sigmoid", 4), ("Dance", "visual_osn_fc3_2ReLUSigmoid", 4)],
        ["box_xywh_sim", "OSNet_sim"],
    ),
    "pt21_dynamic_iou_oks_OSNet": (
        "test_dl_pt21_256x192_rcnn_085_040",
        "submission_pt21",
        "dynamic",
        [
            ("pt21", "box_fc1_Sigmoid", 4),
            ("pt21", "pose_coco_fc1_Sigmoid", 4),
            ("pt21", "visual_osn_fc5_4ReLUSigmoid", 4),
        ],
        ["box_xywh_sim", "pose_sim_coco", "OSNet_sim"],
    ),
}


def run(config: Config, dl_key: str) -> None:
    """Main function to run the code."""

    with HidePrint():
        # validation dataset
        val_dl = module_loader(config=config, module_type="dataloader", key=dl_key)

        # will load all the similarity modules
        engine = DGSEngine(config=config, path=["engine"], test_loader=val_dl)
        close_all_layers(engine.model)

    engine.predict()

    # end processes
    engine.terminate()


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")

    cfg = load_config(CONFIG_FILE)

    for NAME, (DL_KEY, SUBMISSION, COMBINE_FLAG, ALPHA, SIM_MODS) in (
        pbar_settings := tqdm(SETTINGS.items(), desc="run configs", leave=False)
    ):
        pbar_settings.set_postfix_str(NAME)

        cfg["name"] = f"Test-DGS-{NAME}"
        cfg["log_dir_suffix"] = NAME
        cfg["test"]["submission"] = SUBMISSION
        cfg["DGSModule"]["names"] = SIM_MODS
        cfg[DL_KEY]["load_img_crops"] = not all(sm.startswith("box") or sm.startswith("pose") for sm in SIM_MODS)

        if COMBINE_FLAG == "dynamic":
            cfg["DGSModule"]["combine"] = "dynamic_combine"
            for i, a_i in enumerate(ALPHA):
                weight_files = glob(f"./weights/trained_alpha/{a_i[0]}/{SIM_MODS[i]}/{a_i[1]}/ep{a_i[2]:03d}_lr*.pth")
                if len(weight_files) < 1:
                    raise FileNotFoundError(f"No weight file found for {a_i}")
                cfg[a_i[1]]["weight"] = weight_files[0]

            cfg["dynamic_combine"]["alpha_modules"] = [a[1] for a in ALPHA]
        elif COMBINE_FLAG == "static":
            cfg["DGSModule"]["combine"] = "static_combine"
            cfg["static_combine"]["alpha"] = ALPHA
        else:
            raise NotImplementedError

        if "pt21" in DL_KEY:
            data_paths = glob(cfg[DL_KEY]["paths"])
        elif "Dance" in DL_KEY:
            data_paths = [os.path.normpath(p) for p in glob(cfg[DL_KEY]["paths"])]
        else:
            raise NotImplementedError
        assert len(data_paths)

        # iterate over all sub datasets
        for sub_datapath in (pbar_data := tqdm(data_paths, desc="ds_sub_dir", leave=False)):
            if "pt21" in DL_KEY:
                ds_name = os.path.basename(sub_datapath)
            elif "Dance" in DL_KEY:
                ds_name = os.path.basename(os.path.normpath(os.path.dirname(os.path.dirname(sub_datapath))))
            else:
                raise NotImplementedError
            pbar_data.set_postfix_str(ds_name)

            cfg["test"]["writer_log_dir_suffix"] = f"./{ds_name}/"
            cfg[DL_KEY]["data_path"] = sub_datapath

            # set the new path for the out file in the log_dir
            if "pt21" in DL_KEY:
                cfg[SUBMISSION]["file"] = os.path.abspath(
                    os.path.normpath(os.path.join(cfg["log_dir"], NAME, f"./results_json/{ds_name}.json"))
                )
            elif "Dance" in DL_KEY:
                cfg[SUBMISSION]["file"] = os.path.abspath(
                    os.path.normpath(os.path.join(cfg[DL_KEY]["dataset_path"], f"./results_{NAME}/{ds_name}.txt"))
                )
            else:
                raise NotImplementedError

            if os.path.exists(cfg[SUBMISSION]["file"]):
                continue

            run(config=cfg, dl_key=DL_KEY)
