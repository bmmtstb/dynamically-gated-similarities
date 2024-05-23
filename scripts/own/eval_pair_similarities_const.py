"""
Use the DGS module to evaluate the performance of a combination of two similarities.

Every pair of similarities is evaluated using the DGS module on its own with constant alpha values.
Every combination of alpha values in 10% increments is used.

1. [10, 90]
2. [20, 80]
...
9. [90, 10]

The evaluation is run over the ground-truth evaluation set of the PT21 dataset,
and additionally over the PT21 evaluation set but using the RCNN-dataloader to obtain the predictions.
"""

# pylint: disable=R0801

import os

import torch
from tqdm import tqdm

from dgs.models.dgs import DGSModule
from dgs.models.engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers
from dgs.utils.types import Config
from dgs.utils.utils import HidePrint

CONFIG_FILE = "./configs/DGS/eval_const_pairwise_similarities.yaml"


# @torch_memory_analysis
# @MemoryTracker(interval=7.5, top_n=20)
@torch.no_grad()
def run(config: Config, dl_key: str, paths: list[str]) -> None:
    """Main function to run the code."""
    # Combinations of IoU, OKS, and visual similarity
    for dgs_key in (pbar_key := tqdm(["iou_oks", "iou_OSNet", "oks_OSNet"], desc="combinations")):
        pbar_key.set_postfix_str(dgs_key)

        config["name"] = f"Evaluate-Pairwise-Combinations-{dgs_key}"

        # get sub folders or files and analyse them one-by-one
        for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
            pbar_data.set_postfix_str(os.path.basename(sub_datapath))

            config[dl_key]["data_path"] = sub_datapath

            for alpha_1 in (pbar_alpha := tqdm(range(10, 91, 10), desc="alpha", leave=False)):
                pbar_alpha.set_postfix_str(f"[{alpha_1}, {100 - alpha_1}]")

                # set alpha values
                config["combine_sim"]["alpha"] = [alpha_1 / 100.0, (100.0 - alpha_1) / 100.0]

                # make sure to have a unique log dir every time
                orig_log_dir = config["log_dir"]

                # change config data
                config["log_dir"] += f"./{dl_key}/{dgs_key}_{alpha_1}_{100-alpha_1}/"
                # set the new path for the out file in the log_dir
                config["submission"]["file"] = os.path.abspath(
                    os.path.normpath(
                        f"{config['log_dir']}/results_json/{sub_datapath.split('/')[-1].removesuffix('.json')}.json"
                    )
                )

                if os.path.exists(config["submission"]["file"]):
                    # reset the original log dir
                    config["log_dir"] = orig_log_dir
                    continue

                with HidePrint():
                    # validation dataset
                    val_dl = module_loader(config=config, module_class="dataloader", key=dl_key)

                    # will load all the similarity modules
                    model: DGSModule = module_loader(config=config, module_class="dgs", key=dgs_key).cuda()
                    close_all_layers(model)

                    engine = DGSEngine(config=config, model=model, test_loader=val_dl)

                engine.test()

                # end processes
                engine.terminate()
                del val_dl, model, engine

                # reset the original log dir
                config["log_dir"] = orig_log_dir


if __name__ == "__main__":
    print(f"Cuda available: {torch.cuda.is_available()}")

    print("Evaluating on the PT21 ground-truth evaluation dataset")
    cfg = load_config(CONFIG_FILE)
    base_path = cfg["dgs_gt"]["base_path"]
    data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]
    run(config=cfg, dl_key="dgs_gt", paths=data_paths)

    print("Evaluating on the PT21 eval-dataset using KeypointRCNN as prediction backbone")
    cfg = load_config(CONFIG_FILE)
    base_path = cfg["dgs_rcnn"]["base_path"]
    data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]
    run(config=cfg, dl_key="dgs_rcnn", paths=data_paths)
