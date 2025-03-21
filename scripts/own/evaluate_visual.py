"""
Given a folder containing model weights or checkpoints, evaluate them using a single DGS model.
"""

import glob
import os
import time
from datetime import timedelta

import torch as t
from tqdm import tqdm

from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.files import to_abspath
from dgs.utils.utils import HidePrint

CONFIG_FILE = "./configs/eval_visual.yaml"
SUB_DIR = "./results/own/visual_sim/20240210/checkpoints/Visual_Similarity_(Own)_0.0003/"

if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)

    print(f"Cuda available: {t.cuda.is_available()}")

    ds_start_time = time.time()
    # test / gallery
    test_dl = module_loader(config=config, module_type="dataloader", key="test_dl")
    # validation / query
    val_dl = module_loader(config=config, module_type="dataloader", key="val_dl")

    print(f"Total dataset loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    with HidePrint():
        model = module_loader(config=config, module_type="similarity", key="visual_similarity").cuda()
        model.eval()

    engine = module_loader(
        config=config, module_type="engine", key="engine", model=model, test_loader=test_dl, val_loader=val_dl
    )

    model_paths = sorted(glob.glob(os.path.join(to_abspath(SUB_DIR), "*.pth")), reverse=True)

    for checkpoint_file in tqdm(model_paths, desc="Checkpoints", position=0):
        # load checkpoint - set engine parameters and load model weights
        engine.load_model(path=checkpoint_file)
        # test / evaluate the module with the loaded weights
        engine.test()
