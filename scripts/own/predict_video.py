"""Use the (trained) DGS module to track / predict a video input."""

import os
import time
from datetime import timedelta

import torch

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import fill_in_defaults, load_config
from dgs.utils.utils import HidePrint

CONFIG_FILE = "./configs/predict_video.yaml"


if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = fill_in_defaults(load_config(CONFIG_FILE))
    print(f"Cuda available: {torch.cuda.is_available()}")

    # validation dataset
    print("Loading Data(set)")
    ds_start_time = time.time()
    test_dl = module_loader(config=config, module_class="dataloader", key="dataloader_test")
    print(f"Total data(set) loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    module_start_time = time.time()
    with HidePrint():
        # will load all the similarity modules
        model: DGSModule = module_loader(config=config, module_class="dgs", key="dgs").cuda()
        model.eval()
    print(f"Total model loading time: {str(timedelta(seconds=round(time.time() - module_start_time)))}")

    engine = DGSEngine(config=config, model=model, test_loader=test_dl)

    engine.predict()

    print("Combine images to video")
    # vid_file = os.path.abspath(os.path.join(model.log_dir, "./prediction.mp4"))
    # combine_images_to_video(
    #     imgs=os.path.abspath(os.path.join(model.log_dir, "./images/")),
    #     video_file=vid_file,
    # )
    os.system("/bin/bash ffmpeg -framerate 30 -pattern_type glob -i './images/*.png' out.mp4")
    # print(f"Video available at: {vid_file}")
