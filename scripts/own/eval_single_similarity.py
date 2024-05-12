"""
Use the DGS module to evaluate the performance of the individual similarities.

Every similarity is evaluated using the DGS module on its own with an alpha value of 1.

The evaluation is run over the ground-truth evaluation set of the PT21 dataset,
and additionally over the PT21 evaluation set but using the RCNN-dataloader to obtain the predictions.
"""

import os

import torch
from tqdm import tqdm

from dgs.models.dgs.dgs import DGSModule
from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers, memory_analysis
from dgs.utils.utils import HidePrint

CONFIG_FILE = "./configs/DGS/eval_sim_indep.yaml"


@memory_analysis
@torch.no_grad()
def run():
    """Main function to run the code."""
    print(f"Loading configuration: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)

    # first eval using GT data
    # second eval using RCNN data
    for dl_key in ["dl_gt", "dl_rcnn"]:
        print(f"Evaluating on the {dl_key} dataloader")

        # get dataset folder (one by one)
        base_path = config[dl_key]["base_path"]
        dataset_folders = [f.path for f in os.scandir(base_path) if f.is_dir()]

        for sub_folder in tqdm(dataset_folders, desc="ds_sub_dir"):

            config[dl_key]["data_path"] = sub_folder

            # IoU, OKS, and visual similarity
            for dgs_key in tqdm(["dgs_box", "dgs_pose", "dgs_vis_3", "dgs_vis_2", "dgs_vis_3"], desc="sims"):

                # make sure to have a unique log dir every time
                orig_log_dir = config["log_dir"]
                config["log_dir"] += f"./{dl_key}/{dgs_key}/"

                with HidePrint():
                    # validation dataset
                    val_dl = module_loader(config=config, module_class="dataloader", key=dl_key)

                    # will load all the similarity modules
                    model: DGSModule = module_loader(config=config, module_class="dgs", key=dgs_key).cuda()
                    close_all_layers(model)

                    engine = DGSEngine(config=config, model=model, test_loader=val_dl)

                engine.test()

                # reset the original log dir
                config["log_dir"] = orig_log_dir


if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}")
    run()
