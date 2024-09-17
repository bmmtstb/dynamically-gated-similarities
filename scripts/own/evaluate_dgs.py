"""
Evaluate the DGS Tracker.
"""

import time
from datetime import timedelta

import torch as t

from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.utils import HidePrint

CONFIG_FILE = "./configs/eval_dgs.yaml"

if __name__ == "__main__":
    print(f"Loading configuration: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)
    print(f"Cuda available: {t.cuda.is_available()}")

    # validation dataset
    ds_start_time = time.time()
    test_dl = module_loader(config=config, module_type="dataloader", key="dataloader_test")
    print(f"Total dataset loading time: {str(timedelta(seconds=round(time.time() - ds_start_time)))}")

    module_start_time = time.time()
    with HidePrint():
        # will load all the similarity modules
        model = module_loader(config=config, module_type="dgs", key="dgs").cuda()
        model.eval()
    print(f"Total model loading time: {str(timedelta(seconds=round(time.time() - module_start_time)))}")

    engine = DGSEngine(config=config, model=model, test_loader=test_dl)

    engine.test()
