"""Use the (trained) DGS module to track / predict a video input."""

import time
from datetime import timedelta

import torch as t

from dgs.models.dgs.dgs import DGSModule
from dgs.models.loader import module_loader
from dgs.utils.config import load_config

# from dgs.utils.utils import HidePrint

CONFIG_FILE = "./configs/predict_images.yaml"

if __name__ == "__main__":
    print(f"Loading the configuration: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)
    print(f"Cuda available: {t.cuda.is_available()}")

    # validation dataset
    print("Loading datasets and data loaders")
    test_dl = module_loader(config=config, module_class="dataloader", key="dataloader_test")

    module_start_time = time.time()
    # some submodules (especially the torchreid ones) print lots of debug info during model loading.
    # You can hide those outputs using the HidePrint context manager
    # with HidePrint():

    # The overall DGS Module will load all the similarity modules
    model: DGSModule = module_loader(config=config, module_class="dgs", key="dgs").cuda()
    model.eval()
    print(f"Total model loading time: {str(timedelta(seconds=round(time.time() - module_start_time)))}")

    # Use module_loader to load the engine, make sure to pass the required engine kwargs as additional kwargs
    engine = module_loader(config=config, module_class="engine", key="dgs_engine", model=model, test_dl=test_dl)

    # Use the engine to predict / track the given data (test_dl)
    engine.predict()

    # Use the data according to your needs
    print(f"The predicted images and the predicted key points, bboxes, and tracks are saved in '{model.log_dir}'")
    print("\n---\n")
    print("To combine the predicted images to a video file we can use the 'ffmpeg' library. Run: \n")
    print(f"cd {model.log_dir}")
    print("ffmpeg -framerate 30 -pattern_type glob -i './images/*.png' prediction.mp4")
