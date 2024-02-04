"""
Train a pose embedding model and predict pose similarities using the dgs internal engine.
"""

import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, Dataset

from dgs.models.dataset import get_data_loader
from dgs.models.engine import VisualSimilarityEngine
from dgs.models.loader import module_loader
from dgs.utils.config import fill_in_defaults, load_config

CONFIG_FILE = "./configs/train_visual_similarity.yaml"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
TEST_ONLY = True


if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = fill_in_defaults(load_config(CONFIG_FILE))

    print(f"Cuda available: {torch.cuda.is_available()}")

    ds_start_time = time.time()
    # test / gallery
    test_ds: Dataset = module_loader(config=config, module="dataset_test")
    test_dl: DataLoader = get_data_loader(ds=test_ds, batch_size=TEST_BATCH_SIZE)
    # validation / query
    val_ds: Dataset = module_loader(config=config, module="dataset_valid")
    val_dl: DataLoader = get_data_loader(ds=val_ds, batch_size=TEST_BATCH_SIZE)

    if TEST_ONLY:
        train_dl = None  # pylint: disable=invalid-name
    else:
        train_ds: Dataset = module_loader(config=config, module="dataset_train")
        train_dl = get_data_loader(ds=train_ds, batch_size=TRAIN_BATCH_SIZE)

    print(f"Total dataset loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    model = module_loader(config=config, module="embedding_generator_visual")

    engine = VisualSimilarityEngine(
        config=config,
        model=model,
        test_loader=test_dl,
        val_loader=val_dl,
        train_loader=train_dl,
        test_only=TEST_ONLY,
    )

    # train and test the module
    engine.run()
