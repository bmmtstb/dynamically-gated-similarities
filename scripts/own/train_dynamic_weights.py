"""
Train, evaluate, and test the dynamic weights with different individual weights on the |PT21|_ and |Dance|_ datasets.
"""

# pylint: disable=R0801

import os
from glob import glob
from typing import Union

import torch as t
from torch import nn
from tqdm import tqdm

from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.nn import fc_linear
from dgs.utils.torchtools import init_model_params
from dgs.utils.types import Config
from dgs.utils.utils import notify_on_completion_or_error, send_discord_notification

CONFIG_FILE = "./configs/DGS/train_dynamic_weights.yaml"

DL_KEYS: list[tuple[str, str, str]] = [
    # DanceTrack with evaluation using the accuracy of the weights
    ("train_dl_dance_256x192", "val_dl_dance_256x192_eval_acc", "test_dl_dance_256x192"),
    # DanceTrack with evaluation using the MOTA and HOTA metrics
    # ("train_dl_dance_256x192", "val_dl_dance_256x192", "test_dl_dance_256x192"),
]

ALPHA_MODULES: dict[str, Union[nn.Module, nn.Sequential]] = {
    "fc_1_box": fc_linear(hidden_layers=[4, 1]),
    "sig_fc_1_box": nn.Sequential(nn.Sigmoid(), fc_linear(hidden_layers=[4, 1])),
    "fc_1_box_sig": nn.Sequential(fc_linear(hidden_layers=[4, 1]), nn.Sigmoid()),
    "fc_1_pose_coco": nn.Sequential(
        nn.Flatten(),
        fc_linear(hidden_layers=[17, 1]),
    ),
    "conv_1_fc_1_pose_coco": nn.Sequential(
        nn.Conv2d(17, 17, kernel_size=(2, 5), bias=True),
        nn.Flatten(),
        fc_linear(hidden_layers=[17, 1]),
    ),
    "fc_1_visual": fc_linear([512, 1]),
    "fc_1_visual_sig": nn.Sequential(fc_linear([512, 1]), nn.Sigmoid()),
    "fc_2_visual": fc_linear([512, 128, 1]),
    "fc_3_visual": fc_linear([512, 256, 128, 1]),
    "fc_4_visual": fc_linear([512, 256, 128, 64, 1]),
    "fc_5_visual": fc_linear([512, 256, 128, 64, 32, 1]),
}

NAMES: dict[str, list[str]] = {
    "box_sim": ["fc_1_box", "fc_1_box_sig", "sig_fc_1_box"],
    # "pose_sim_coco": ["fc_1_pose_coco", "conv_1_fc_1_pose_coco"],
    "OSNet_sim": ["fc_1_visual", "fc_1_visual_sig", "fc_2_visual"],
    # "OSNetAIN_sim": ["fc_1_visual", "fc_2_visual"],
    # "Resnet50_sim": ["fc_1_visual", "fc_2_visual"],
    # "Resnet152_sim": ["fc_1_visual", "fc_2_visual"],
}


# @torch_memory_analysis
# @MemoryTracker(interval=7.5, top_n=20)
@notify_on_completion_or_error(min_time=30, info="run initial weight")
@t.no_grad()
def test_pt21(cfg: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
    """Set the PT21 config."""
    crop_h, crop_w = cfg[dl_key]["crop_size"]
    cfg[dl_key]["crops_folder"] = (
        cfg[dl_key]["base_path"]
        .replace("posetrack_data", f"crops/{crop_h}x{crop_w}")
        .replace(f"{crop_h}x{crop_w}_", "")  # remove redundant from crop folder name iff existing
    )

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        pbar_data.set_postfix_str(os.path.basename(sub_datapath))
        # make sure to have a unique log dir every time
        orig_log_dir = cfg["log_dir"]

        # change config data
        cfg[dl_key]["data_path"] = sub_datapath
        cfg["log_dir"] += f"./{out_key}/{dgs_key}/"
        cfg["test"]["submission"] = ["submission_pt21"]

        # set the new path for the out file in the log_dir
        subm_key = "submission_pt21"
        cfg[subm_key]["file"] = os.path.abspath(
            os.path.normpath(f"{cfg['log_dir']}/results_json/{sub_datapath.split('/')[-1].removesuffix('.json')}.json")
        )

        if os.path.exists(cfg[subm_key]["file"]):
            # reset the original log dir
            cfg["log_dir"] = orig_log_dir
            continue

        engine = get_dgs_engine(cfg=cfg, dl_keys=(None, None, dl_key), dgs_key=dgs_key)
        engine.test()
        # end processes
        engine.terminate()

        # reset the original log dir
        cfg["log_dir"] = orig_log_dir


# @torch_memory_analysis
@notify_on_completion_or_error(min_time=30, info="run initial weight")
@t.no_grad()
def test_dance(cfg: Config, dl_key: str, paths: list, out_key: str, dgs_key: str) -> None:
    """Set the DanceTrack config."""

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        dataset_path = os.path.normpath(os.path.dirname(os.path.dirname(sub_datapath)))
        dataset_name = os.path.basename(dataset_path)
        pbar_data.set_postfix_str(dataset_name)
        cfg[dl_key]["data_path"] = sub_datapath

        # make sure to have a unique log dir every time
        orig_log_dir = cfg["log_dir"]

        # change config data
        cfg["log_dir"] += f"./{out_key}/{dgs_key}/"
        cfg["test"]["writer_log_dir_suffix"] = f"./{os.path.basename(sub_datapath)}/"

        # set the new path for the submission file
        subm_key = "submission_MOT"
        cfg["test"]["submission"] = [subm_key]
        cfg[subm_key]["file"] = os.path.abspath(
            os.path.normpath(f"{os.path.dirname(dataset_path)}./results_{out_key}_{dgs_key}/{dataset_name}.txt")
        )

        if os.path.exists(cfg[subm_key]["file"]):
            # reset the original log dir
            cfg["log_dir"] = orig_log_dir
            continue

        engine = get_dgs_engine(cfg=cfg, dl_keys=(None, None, dl_key), dgs_key=dgs_key)
        engine.test()
        # end processes
        engine.terminate()

        # reset the original log dir
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
        "model": module_loader(config=cfg, module_class="dgs", key=dgs_key),
    }
    # validation dataset
    if key_train is not None:
        kwargs["train_loader"] = module_loader(config=cfg, module_class="dataloader", key=key_train)
    if key_eval is not None:
        kwargs["val_loader"] = module_loader(config=cfg, module_class="dataloader", key=key_eval)
    if key_test is not None:
        kwargs["test_loader"] = module_loader(config=cfg, module_class="dataloader", key=key_test)

    return module_loader(config=cfg, module_class="engine", key=engine_key, **kwargs)


def train_dgs_engine(cfg: Config, dl_train_key: str, dl_eval_key: str, alpha_mod_name: str, sim_name: str) -> None:
    """Train and evaluate the DGS engine."""
    _crop_h, _crop_w = cfg[dl_train_key]["crop_size"]
    lr = cfg["train"]["optimizer_kwargs"]["lr"]

    # modify config
    cfg["DGSModule"]["names"] = [sim_name]
    cfg["log_dir_suffix"] = f"./{dl_train_key}/{sim_name}/{alpha_mod_name}_{lr:.10f}/"

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

    engine_train.train_model()
    engine_train.terminate()
    send_discord_notification(f"finished training and evaluation of {sim_name} - {alpha_mod_name} - {dl_train_key}")


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")

    config = load_config(CONFIG_FILE)

    # for every similarity or combination of similarities
    for SIM_NAME, alpha_modules in (pbar_key := tqdm(NAMES.items(), desc="similarities")):
        pbar_key.set_postfix_str(SIM_NAME)

        for ALPHA_MOD_NAME in (pbar_alpha_mod := tqdm(alpha_modules, desc="alpha_modules", leave=False)):
            pbar_alpha_mod.set_postfix_str(ALPHA_MOD_NAME)

            # set name
            config["name"] = f"Train-Dynamic-Weights-Individually-{SIM_NAME}-{ALPHA_MOD_NAME}"

            for DL_KEY in DL_KEYS:
                DL_TRAIN_KEY, DL_EVAL_KEY, DL_TEST_KEY = DL_KEY

                # ##################### #
                # TRAINING & EVALUATION #
                # ##################### #
                train_dgs_engine(
                    cfg=config,
                    dl_train_key=DL_TRAIN_KEY,
                    dl_eval_key=DL_EVAL_KEY,
                    alpha_mod_name=ALPHA_MOD_NAME,
                    sim_name=SIM_NAME,
                )

                # ####### #
                # TESTING #
                # ####### #

                # TODO load weights from the best model and test them or from a list of given names, ...
                # print(f"Testing on the rcnn predictions of the test-dataset: {SIM_NAME} - {DL_TEST}")
                # if "pt21" in DL_TEST:
                #     test_pt21(
                #         config=cfg,
                #         dl_key=DL_TEST,
                #         paths=[os.path.normpath(f) for f in glob(cfg[DL_TEST]["paths"])],
                #         out_key=f"{DL_TEST}_{alpha_module}",
                #         dgs_key=SIM_NAME,
                #     )
                # elif "Dance" in DL_TEST:
                #     test_dance(
                #         config=cfg,
                #         dl_key=DL_TEST,
                #         paths=[os.path.normpath(f) for f in glob(cfg[DL_TEST]["paths"])],
                #         out_key=f"{DL_TEST}_{alpha_module}",
                #         dgs_key=SIM_NAME,
                #     )
                # else:
                #     raise NotImplementedError