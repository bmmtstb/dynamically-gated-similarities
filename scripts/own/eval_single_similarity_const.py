"""
Use the DGS module to evaluate the performance of the individual similarities.

Every similarity is evaluated using the DGS module on its own with an alpha value of 1.

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

CONFIG_FILE = "./configs/DGS/eval_const_single_similarities.yaml"


# @torch_memory_analysis
# @MemoryTracker(interval=7.5, top_n=20)
@torch.no_grad()
def run(config: Config, dl_key: str, paths: list[str], out_key: str) -> None:
    """Main function to run the code."""
    # IoU, OKS, and visual similarity
    for dgs_key in (
        pbar_key := tqdm(
            [
                "iou",
                "oks",
                "OSNet",
                "OSNetAIN",
                "Resnet50",
                "Resnet152",
                "OSNetAIN_CrossDomainDuke",
                "OSNetIBN_CrossDomainDuke",
                "OSNetAIN_CrossDomainMSMT17",
            ],
            desc="similarities",
        )
    ):
        pbar_key.set_postfix_str(dgs_key)

        config["name"] = f"Evaluate-Single-{dgs_key}"

        # get sub folders or files and analyse them one-by-one
        for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
            pbar_data.set_postfix_str(os.path.basename(sub_datapath))
            # make sure to have a unique log dir every time
            orig_log_dir = config["log_dir"]

            # change config data
            config[dl_key]["data_path"] = sub_datapath
            config["log_dir"] += f"./{out_key}/{dgs_key}/"
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
    run(config=cfg, dl_key="dgs_gt", paths=data_paths, out_key="dgs_gt")

    print("Evaluating on the PT21 eval-dataset using KeypointRCNN as prediction backbone")
    # for thresh in (pbar_thresh := tqdm(["085", "090", "095", "099"], desc="thresholds")):
    for score_thresh in (pbar_score_thresh := tqdm(["085", "090", "095"], desc="Score Thresh")):
        pbar_score_thresh.set_postfix_str(os.path.basename(score_thresh))
        for iou_thresh in (pbar_iou_thresh := tqdm(["030", "050", "070", "090"], desc="IoU Thresh")):
            pbar_iou_thresh.set_postfix_str(os.path.basename(iou_thresh))

            cfg = load_config(CONFIG_FILE)
            base_path = f"./data/PoseTrack21/posetrack_data/rcnn_prediction_{score_thresh}_{iou_thresh}/"
            cfg["dgs_rcnn"]["base_path"] = base_path
            cfg["dgs_rcnn"][
                "crops_folder"
            ] = f"./data/PoseTrack21/crops/256x192/rcnn_prediction_{score_thresh}_{iou_thresh}/"
            data_paths = [f.path for f in os.scandir(base_path) if f.is_file()]
            run(config=cfg, dl_key="dgs_rcnn", paths=data_paths, out_key=f"dgs_rcnn_{score_thresh}_{iou_thresh}")