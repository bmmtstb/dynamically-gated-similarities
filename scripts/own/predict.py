"""Use the (trained) DGS module to track / predict a video input."""

import time
from datetime import timedelta

import torch as t

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.utils import HidePrint

# CONFIG_FILE = "./configs/predict_video.yaml"
CONFIG_FILE = "./configs/predict_images.yaml"


if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)
    print(f"Cuda available: {t.cuda.is_available()}")

    # validation dataset
    print("Loading data")
    ds_start_time = time.time()
    test_dl = module_loader(config=config, module_type="dataloader", key="dataloader_test")
    print(f"Total data loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    module_start_time = time.time()
    with HidePrint():
        # will load all the similarity modules
        model: DGSModule = module_loader(config=config, module_type="dgs", key="dgs").cuda()
        model.eval()
    print(f"Total model loading time: {str(timedelta(seconds=round(time.time() - module_start_time)))}")

    engine = DGSEngine(config=config, model=model, test_loader=test_dl)

    engine.predict()

    print("Combine images to video")
    print("Use ffmpeg, because it is faster and more stable. Run:")
    print(f"cd {model.log_dir}")
    print("ffmpeg -framerate 30 -pattern_type glob -i './images/*.png' prediction.mp4")
    print("----")
    print(
        "Or use function 'combine_images_to_video' as commented out in 'scripts/predict.py', "
        "but this function is way slower."
    )
    # vid_file = os.path.abspath(os.path.join(model.log_dir, "./prediction.mp4"))
    # combine_images_to_video(
    #     imgs=os.path.abspath(os.path.join(model.log_dir, "./images/")),
    #     video_file=vid_file,
    # )
