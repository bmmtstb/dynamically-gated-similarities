"""
Given a folder containing model weights or checkpoints, evaluate them.
"""

import glob
import os
import time
from datetime import timedelta

import torch
from tqdm import tqdm

from dgs.models.engine import VisualSimilarityEngine
from dgs.models.loader import module_loader
from dgs.utils.config import fill_in_defaults, load_config
from dgs.utils.torchtools import resume_from_checkpoint

CONFIG_FILE = "./configs/eval_visual.yaml"
SUB_DIR = "./20240210/checkpoints/Visual_Similarity_(Own)_0.0003/"

if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = fill_in_defaults(load_config(CONFIG_FILE))

    print(f"Cuda available: {torch.cuda.is_available()}")

    ds_start_time = time.time()
    # test / gallery
    test_dl = module_loader(config=config, module="dataloader_test")
    # validation / query
    val_dl = module_loader(config=config, module="dataloader_valid")

    print(f"Total dataset loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    model = module_loader(config=config, module="embedding_generator_visual").cuda().eval()

    engine = VisualSimilarityEngine(
        config=config,
        model=model,
        test_loader=test_dl,
        val_loader=val_dl,
    )

    for checkpoint_file in tqdm(glob.glob(os.path.join(SUB_DIR, "*.pth")), desc="Checkpoints"):
        print(f"Loading checkpoint: {checkpoint_file}")
        # load checkpoint - set engine parameters and load model weights
        resume_from_checkpoint(fpath=checkpoint_file, model=model, verbose=True)
        # test / evaluate the module with the loaded weights
        engine.test()
