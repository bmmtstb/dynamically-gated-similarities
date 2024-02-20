"""
Train a pose embedding model and predict pose similarities using the dgs internal engine.
"""

import time
from datetime import timedelta

import torch

from dgs.models import module_loader
from dgs.models.engine import VisualSimilarityEngine
from dgs.utils.config import fill_in_defaults, load_config
from dgs.utils.torchtools import open_all_layers, open_specified_layers

CONFIG_FILE = "./configs/train_visual.yaml"
OPEN_CLASSIFIER_ONLY = True


if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = fill_in_defaults(load_config(CONFIG_FILE))

    print(f"Cuda available: {torch.cuda.is_available()}")

    ds_start_time = time.time()
    # test / gallery
    test_dl = module_loader(config=config, module_name="dataloader_gallery")
    # validation / query
    val_dl = module_loader(config=config, module_name="dataloader_query")
    # train
    train_dl = module_loader(config=config, module_name="dataloader_train")

    print(f"Total dataset loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    model = module_loader(config=config, module_name="embedding_generator_visual")
    # only modify the classifier
    if OPEN_CLASSIFIER_ONLY:
        open_specified_layers(model=model, open_layers=["classifier"], verbose=True)
    else:
        open_all_layers(model)

    engine = VisualSimilarityEngine(
        config=config,
        model=model,
        test_loader=test_dl,
        val_loader=val_dl,
        train_loader=train_dl,
    )

    # train and test the module
    engine.run()
