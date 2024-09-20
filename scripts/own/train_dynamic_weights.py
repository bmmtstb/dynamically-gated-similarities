"""
Train, evaluate, and test the dynamic weights with different individual weights on the |PT21|_ and |Dance|_ datasets.
"""

# pylint: disable=R0801

import os
import subprocess
from glob import glob
from typing import Union

import torch as t
from torch import nn
from tqdm import tqdm

from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.nn import fc_linear
from dgs.utils.torchtools import close_all_layers, init_model_params
from dgs.utils.types import Config
from dgs.utils.utils import notify_on_completion_or_error

CONFIG_FILE = "./configs/DGS/train_dynamic_weights.yaml"

DL_KEYS_TRAIN: list[tuple[str, str]] = [
    # DanceTrack with evaluation using the accuracy of the weights
    # ("train_dl_dance_256x192", "val_dl_dance_256x192_eval_acc"),
    # DanceTrack with evaluation using the MOTA and HOTA metrics
    # ("train_dl_dance_256x192", "val_dl_dance_256x192"),
    ("train_dl_pt21_256x192", "val_dl_pt21_256x192_eval_acc"),
]

DL_KEYS_EVAL: dict[str, dict[str, list[tuple[str, int]]]] = {
    # dance gt
    "val_dl_dance_256x192": {
        "box_sim": [
            ("fc_1_box", 4),
            ("fc_1_box", 6),
            # ("fc_2_box", 6),
        ],
        "OSNet_sim": [
            # ("fc_1_visual", 6),
            # ("fc_2_visual", 6),
            # ("fc_3_visual", 6),
            # ("fc_4_visual", 6),
            # ("fc_5_visual", 6),
        ],
    },
    # pt21 gt
    "val_dl_pt21_256x192": {
        "box_sim": [
            # ("fc_1_box", 4),
            # ("fc_1_box", 6),
            # ("fc_2_box", 6),
        ],
        "pose_sim_coco": [
            # ("fc_1_pose_coco", 6),
            # ("conv1_o15k2_fc_1_pose_coco", 6),
        ],
        "OSNet_sim": [
            # ("fc_1_visual", 6),
            # ("fc_2_visual", 6),
            # ("fc_3_visual", 6),
            # ("fc_4_visual", 6),
            # ("fc_5_visual", 6),
        ],
    },
}

ALPHA_MODULES: dict[str, Union[nn.Module, nn.Sequential]] = {
    "box_fc_1": fc_linear(hidden_layers=[4, 1]),
    "box_fc_2": fc_linear(hidden_layers=[4, 8, 1]),
    "pose_coco_fc_1": nn.Sequential(
        nn.Flatten(),
        fc_linear(hidden_layers=[2 * 17, 1]),
    ),
    "pose_cococonv1_o15k2_fc_1": nn.Sequential(
        nn.Conv1d(17, 15, kernel_size=2, groups=1, bias=True),
        nn.Flatten(),
        fc_linear(hidden_layers=[15, 1]),
    ),
    "visual_osn_fc_1": fc_linear([512, 1]),
    "visual_osn_fc_2": fc_linear([512, 128, 1]),
    "visual_osn_fc_3": fc_linear([512, 256, 128, 1]),
    "visual_osn_fc_4": fc_linear([512, 256, 128, 64, 1]),
    "visual_osn_fc_5": fc_linear([512, 256, 128, 64, 32, 1]),
    "visual_res_fc_1": fc_linear([2048, 1]),
    "visual_res_fc_2": fc_linear([2048, 512, 1]),
    "visual_res_fc_3": fc_linear([2048, 512, 64, 1]),
    "visual_res_fc_4": fc_linear([2048, 1024, 256, 64, 1]),
    "visual_res_fc_5": fc_linear([2048, 1024, 256, 128, 64, 1]),
}

NAMES: dict[str, list[str]] = {
    "box_sim": [
        "fc_1_box",
        "fc_2_box",
    ],
    "pose_sim_coco": [
        "fc_1_pose_coco",
        "conv1_o15k2_fc_1_pose_coco",
    ],
    "OSNet_sim": [
        "visual_osn_fc_1",
        # "visual_osn_fc_2",
        "visual_osn_fc_3",
        # "visual_osn_fc_4",
        "visual_osn_fc_5",
    ],
    "OSNetAIN_sim": [
        "visual_osn_fc_1",
        # "visual_osn_fc_2",
        "visual_osn_fc_3",
    ],
    "Resnet50_sim": [
        "visual_res_fc_1",
        # "visual_res_fc_2",
        "visual_res_fc_3",
        # "visual_res_fc_4",
        "visual_res_fc_5",
    ],
    "Resnet152_sim": [
        "visual_res_fc_1",
        # "visual_res_fc_2",
        "visual_res_fc_3",
    ],
}


@notify_on_completion_or_error(min_time=30)
@t.no_grad()
def test_pt21_dynamic_alpha(
    cfg: Config, dl_key: str, paths: list, alpha_mod_name: str, sim_name: str, epoch: int
) -> None:
    """Set the PT21 config."""
    lr = cfg["train"]["optimizer_kwargs"]["lr"]
    subm_key = "submission_pt21"
    folder_path = f"./{dl_key}/{sim_name}/{alpha_mod_name}_{lr:.10f}/"

    # change config that is the same for all sub-datasets (videos) of pt21
    cfg["log_dir_suffix"] = os.path.join(folder_path, f"./test_log/")
    cfg["test"]["submission"] = [subm_key]

    # checkpoint file
    checkpoints = glob(os.path.join(cfg["log_dir"], folder_path, f"./checkpoints/*_epoch{epoch:0>3}.pth"))
    assert len(checkpoints) == 1
    checkpoint_file = os.path.abspath(os.path.normpath(checkpoints[0]))

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        pbar_data.set_postfix_str(os.path.basename(sub_datapath))
        sub_ds_name: str = sub_datapath.split("/")[-1].removesuffix(".json")

        # change config data that is different for each of the sub-datasets (videos) of pt21
        cfg[dl_key]["data_path"] = sub_datapath
        cfg["engine"]["writer_log_dir_suffix"] = f"./{sub_ds_name}/"

        # set the new path for the out file in the log_dir
        cfg[subm_key]["file"] = os.path.abspath(
            os.path.normpath(os.path.join(cfg["log_dir"], folder_path, f"./results_ep{epoch:0>3}/{sub_ds_name}.json"))
        )

        if os.path.exists(cfg[subm_key]["file"]):
            continue

        engine = get_dgs_engine(cfg=cfg, dl_keys=(None, None, dl_key))
        # load weights and freeze layers
        engine.load_model(path=checkpoint_file)
        close_all_layers(engine.model)

        engine.test()

        # terminate and reset log dir
        engine.terminate()


# @torch_memory_analysis
@notify_on_completion_or_error(min_time=30)
@t.no_grad()
def test_dance_dynamic_alpha(
    cfg: Config, dl_key: str, paths: list, alpha_mod_name: str, sim_name: str, epoch: int
) -> None:
    """Set the DanceTrack config."""
    orig_log_dir = cfg["log_dir"]
    subm_key = "submission_MOT"
    lr = cfg["train"]["optimizer_kwargs"]["lr"]
    log_dir = os.path.join(
        cfg[dl_key]["dataset_path"], f"./results_train_{dl_key}_ep{epoch:0>3}/{sim_name}/{alpha_mod_name}_{lr:.10f}/"
    )

    cfg["log_dir"] = log_dir
    cfg["test"]["submission"] = [subm_key]
    cfg["log_dir_suffix"] = "./test_log/"

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        dataset_path = os.path.normpath(os.path.dirname(os.path.dirname(sub_datapath)))
        dataset_name = os.path.basename(dataset_path)
        pbar_data.set_postfix_str(dataset_name)
        cfg[dl_key]["data_path"] = sub_datapath

        # change config data
        cfg["engine"]["writer_log_dir_suffix"] = f"./{dataset_name}/"
        cfg[subm_key]["file"] = os.path.abspath(os.path.normpath(os.path.join(log_dir, f"{dataset_name}.txt")))

        if os.path.exists(cfg[subm_key]["file"]):
            continue

        engine = get_dgs_engine(cfg=cfg, dl_keys=(None, None, dl_key))
        engine.test()
        # end processes
        engine.terminate()
    cfg["log_dir"] = orig_log_dir


@t.no_grad()
def get_dgs_engine(
    cfg: Config,
    dl_keys: tuple[str | None, str | None, str | None],
    engine_key: str = "engine",
    dgs_key: str = "DGSModule",
) -> DGSEngine:
    """Main function to run the code after all the parameters are set."""

    assert any(key is not None for key in dl_keys)
    key_train, key_eval, key_test = dl_keys

    # the DGSModule will load all the similarity modules internally
    kwargs = {
        "model": module_loader(config=cfg, module_type="dgs", key=dgs_key),
    }
    # validation dataset
    if key_train is not None:
        kwargs["train_loader"] = module_loader(config=cfg, module_type="dataloader", key=key_train)
    if key_eval is not None:
        kwargs["val_loader"] = module_loader(config=cfg, module_type="dataloader", key=key_eval)
    if key_test is not None:
        kwargs["test_loader"] = module_loader(config=cfg, module_type="dataloader", key=key_test)

    return module_loader(config=cfg, module_type="engine", key=engine_key, **kwargs)


@notify_on_completion_or_error(min_time=10)
def train_dynamic_alpha(cfg: Config, dl_train_key: str, dl_eval_key: str, alpha_mod_name: str, sim_name: str) -> None:
    """Train and evaluate the DGS engine."""
    _crop_h, _crop_w = cfg[dl_train_key]["crop_size"]
    lr = cfg["train"]["optimizer_kwargs"]["lr"]

    # modify config
    cfg["DGSModule"]["names"] = [sim_name]
    cfg["log_dir_suffix"] = f"./{dl_train_key}/{sim_name}/{alpha_mod_name}_{lr:.10f}/"

    # fixme: add option to resume training in later epochs
    if len(glob(os.path.join(cfg["log_dir"], cfg["log_dir_suffix"], "./checkpoints/lr*_epoch001.pth"))) > 0:
        print(f"return train early: {sim_name} - {alpha_mod_name}")
        return

    # modify config even more
    if "pt21" in dl_train_key:
        cfg["train"]["submission"] = ["submission_pt21"]
    elif "dance" in dl_train_key:
        cfg["train"]["submission"] = ["submission_MOT"]
    else:
        raise NotImplementedError

    cfg["train"]["load_image_crops"] = not any(val in sim_name for val in ["pose_", "box_"])

    print(f"Training on the ground-truth train-dataset with config: {dl_train_key} - {alpha_mod_name}")

    # use the modified config and obtain the model used for training
    engine_train = get_dgs_engine(cfg=cfg, dl_keys=(dl_train_key, dl_eval_key, None))

    # set model and initialize the weights
    engine_train.model.combine.alpha_model = nn.ModuleList([ALPHA_MODULES[alpha_mod_name]])
    engine_train.model.combine.alpha_model.to(device=engine_train.device)
    init_model_params(engine_train.model.combine.alpha_model)

    # train the model(s)
    engine_train.train_model()
    engine_train.terminate()


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")

    config = load_config(CONFIG_FILE)

    # ############################## #
    # TRAINING & Accuracy EVALUATION #
    # ############################## #

    # for every similarity or combination of similarities
    for SIM_NAME, alpha_modules in (pbar_key := tqdm(NAMES.items(), desc="similarities")):
        pbar_key.set_postfix_str(SIM_NAME)

        for ALPHA_MOD_NAME in (pbar_alpha_mod := tqdm(alpha_modules, desc="alpha_modules", leave=False)):
            pbar_alpha_mod.set_postfix_str(ALPHA_MOD_NAME)

            # set name
            config["name"] = f"Train-Dynamic-Weights-Individually-{SIM_NAME}-{ALPHA_MOD_NAME}"

            for DL_KEY in DL_KEYS_TRAIN:
                DL_TRAIN_KEY, DL_EVAL_KEY = DL_KEY

                train_dynamic_alpha(
                    cfg=config,
                    dl_train_key=DL_TRAIN_KEY,
                    dl_eval_key=DL_EVAL_KEY,
                    alpha_mod_name=ALPHA_MOD_NAME,
                    sim_name=SIM_NAME,
                )

    # ########## #
    # EVALUATION #
    # ########## #

    for DL_KEY, similarities in (pbar_dl := tqdm(DL_KEYS_EVAL.items(), desc="datasets")):
        pbar_dl.set_postfix_str(DL_KEY)

        for SIM_NAME, (ALPHA_MOD_NAME, EPOCH) in (
            pbar_key := tqdm(similarities.items(), desc="similarities", leave=False)
        ):
            pbar_key.set_postfix_str(SIM_NAME)

            # TODO load weights from the best model and test them or from a list of given names, ...
            # TODO accept "List_" datasets and test them individually

            if "pt21" in DL_KEY:
                test_pt21_dynamic_alpha(
                    cfg=config,
                    dl_key=DL_KEY,
                    paths=[os.path.normpath(f) for f in glob(config[DL_KEY]["paths"])],
                    alpha_mod_name=ALPHA_MOD_NAME,
                    sim_name=SIM_NAME,
                    epoch=EPOCH,
                )
            elif "Dance" in DL_KEY:
                test_dance_dynamic_alpha(
                    cfg=config,
                    dl_key=DL_KEY,
                    paths=[os.path.normpath(f) for f in glob(config[DL_KEY]["paths"])],
                    alpha_mod_name=ALPHA_MOD_NAME,
                    sim_name=SIM_NAME,
                    epoch=EPOCH,
                )
            else:
                raise NotImplementedError

    # run evaluation steps
    subprocess.call("./scripts/helpers/run_eval_pt21.sh")
    subprocess.call("./scripts/helpers/run_eval_dance.sh")
    exec(open("./scripts/helpers/results_to_csv.py").read())

    # ####### #
    # TESTING #
    # ####### #
