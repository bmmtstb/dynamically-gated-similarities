"""
Given a list of model weights, validate them.
"""

import time
from datetime import timedelta

import torch

from dgs.models.engine import VisualSimilarityEngine
from dgs.models.loader import module_loader
from dgs.utils.config import fill_in_defaults, load_config

CONFIG_FILE = "./configs/train_visual_similarity.yaml"


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

    # train and test the module
    engine.test()
