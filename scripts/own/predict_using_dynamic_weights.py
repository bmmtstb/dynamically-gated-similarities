"""
Use the (trained) DGS module to track / predict a single video input.

Parameters can be changed here and in the configuration file, but most use-cases will only need to change this file.

Make sure that the weights are stored in a folder called ``weights``.
Additionally, either modify the paths or make sure the weights resemble the structure
``./weights/trained_alpha/{dataset_name}/box_xywh_sim/box_fc1_Sigmoid/ep004_lr0_0000100000.pth``
"""

import time
from datetime import timedelta

import torch as t

from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config

CONFIG_FILE = "./configs/DGS/predict_images_trained.yaml"

# DANCE
DL = "dataloader_dance"
SUBMISSION = ["submission_MOT"]
ALPHA_MODULES = ["box_fc2_2Sigmoid", "pose_coco_fc2_2Sigmoid", "visual_osn_fc3_3Sigmoid"]
WEIGHT_DATASET = "Dance"

# PT21
# DL = "dataloader_pt21"
# SUBMISSION = ["submission_pt21"]
# ALPHA_MODULES = ["box_fc2_2Sigmoid", "pose_coco_fc2_2Sigmoid", "visual_osn_fc3_3Sigmoid"]
# WEIGHT_DATASET = "pt21"

if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)
    print(f"Cuda available: {t.cuda.is_available()}")

    # modify config
    config["test"]["submission"] = SUBMISSION
    config["dynamic_alpha_combine"]["alpha_modules"] = ALPHA_MODULES
    for am in ALPHA_MODULES:
        config[am]["weight"] = str(config[am]["weight"]).replace("XXX", WEIGHT_DATASET)

    # validation dataset
    print("Loading data")
    ds_start_time = time.time()
    test_dl = module_loader(config=config, module_type="dataloader", key=DL)
    print(f"Total data loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    module_start_time = time.time()

    engine = DGSEngine(config=config, path=["engine"], test_loader=test_dl)
    engine.model.eval()

    engine.predict()

    print("Combine images to video")
    print("Use ffmpeg, because it is faster and more stable. Run:")
    print(f"cd {engine.log_dir}")
    print("ffmpeg -framerate 30 -pattern_type glob -i './images/*.png' prediction.mp4")
