"""
Train a visual embedding model and visual similarities using the dgs internal engine.
This script might not work and hasn't been tested with later versions of the engine.

Instead of training your own visual embedding generators, you can use the ones provided by |torchreid|_.
"""

import time
from datetime import timedelta

import torch as t

from dgs.models.engine.visual_sim_engine import VisualSimilarityEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers, open_all_layers, open_specified_layers

CONFIG_FILE = "./configs/train_visual.yaml"
OPEN_CLASSIFIER_ONLY = True


if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)

    print(f"Cuda available: {t.cuda.is_available()}")

    ds_start_time = time.time()
    # validation / query
    val_dl = module_loader(config=config, module_type="dataloader", key="query_dl")
    # test / gallery
    test_dl = module_loader(config=config, module_type="dataloader", key="gallery_dl")
    # train
    train_dl = module_loader(config=config, module_type="dataloader", key="train_dl")

    print(f"Total dataset loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    model = module_loader(config=config, module_type="similarity", key="visual_similarity")

    engine = VisualSimilarityEngine(
        config=config,
        model=model,
        test_loader=test_dl,
        val_loader=val_dl,
        train_loader=train_dl,
    )

    # only modify the classifier
    if OPEN_CLASSIFIER_ONLY:
        close_all_layers(engine.model)
        open_specified_layers(model=engine.model, open_layers=["classifier"], verbose=True)
    else:
        open_all_layers(engine.model)

    # train and test the module
    engine.run()
