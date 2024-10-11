"""
Train, evaluate, and test the dynamic weights with different individual weights on the |PT21|_ and |Dance|_ datasets.
"""

# pylint: disable=R0801

import os
from copy import deepcopy
from glob import glob

import torch as t
from tqdm import tqdm

from dgs.models.engine.dgs_engine import DGSEngine
from dgs.models.loader import module_loader
from dgs.utils.config import load_config
from dgs.utils.torchtools import close_all_layers, init_model_params
from dgs.utils.types import Config
from dgs.utils.utils import notify_on_completion_or_error

CONFIG_FILE = "./configs/DGS/train_dynamic_weights.yaml"

TRAIN = True
EVAL = True
TEST = False

DL_KEYS_TRAIN: list[tuple[str, str, dict[str, list[str]]]] = [
    # DanceTrack with evaluation using the accuracy of the weights
    (
        "train_dl_dance_256x192_gt",
        "val_dl_dance_256x192_eval_acc",
        {
            "box_sim": ["box_fc1", "box_fc2"],
            "OSNet_sim": [
                "visual_osn_fc1",
                # "visual_osn_fc2",
                "visual_osn_fc3",
                # "visual_osn_fc4",
                "visual_osn_fc5",
            ],
            "OSNetAIN_sim": [
                # "visual_osn_fc1",
                # "visual_osn_fc2",
                "visual_osn_fc3",
                # "visual_osn_fc4",
                "visual_osn_fc5",
            ],
            "Resnet50_sim": [
                # "visual_res_fc1",
                # "visual_res_fc2",
                "visual_res_fc3",
                # "visual_res_fc4",
                "visual_res_fc5",
            ],
            "Resnet152_sim": [
                # "visual_res_fc1",
                # "visual_res_fc2",
                # "visual_res_fc3",
                # "visual_res_fc4",
                # "visual_res_fc5",
            ],
        },
    ),
    # PoseTrack21 with evaluation using the accuracy of the weights
    (
        "train_dl_pt21_256x192_gt",
        "val_dl_pt21_256x192_eval_acc",
        {
            "box_sim": ["box_fc1", "box_fc2"],
            "pose_sim_coco": [
                # "pose_coco_fc1",
                # "pose_coco_fc2",
                # "pose_coco_conv1o15k2fc1",
                # "pose_coco_conv1o15k2fc2",
            ],
            "OSNet_sim": [
                "visual_osn_fc1",
                # "visual_osn_fc2",
                "visual_osn_fc3",
                # "visual_osn_fc4",
                "visual_osn_fc5",
            ],
            "OSNetAIN_sim": [
                # "visual_osn_fc1",
                # "visual_osn_fc2",
                "visual_osn_fc3",
                # "visual_osn_fc4",
                "visual_osn_fc5",
            ],
            "Resnet50_sim": [
                # "visual_res_fc1",
                # "visual_res_fc2",
                "visual_res_fc3",
                # "visual_res_fc4",
                "visual_res_fc5",
            ],
            "Resnet152_sim": [
                # "visual_res_fc1",
                # "visual_res_fc2",
                # "visual_res_fc3",
                # "visual_res_fc4",
                # "visual_res_fc5",
            ],
        },
    ),
]

DL_KEYS_EVAL: dict[str, dict[str, list[tuple[str, str, int]]]] = {
    # dance gt
    "val_dl_dance_256x192_gt": {
        # earlier model
        "iou_fc1_ep4__vis_fc3_ep4__lr-4": [("box_sim", "box_fc1", 4), ("OSNet_sim", "visual_osn_fc3", 4)],
        # fully trained
        "iou_fc1_ep6__vis_fc1_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc1", 6)],
        "iou_fc1_ep6__vis_fc3_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc3", 6)],
        "iou_fc1_ep6__vis_fc5_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
    },
    "val_dl_dance_256x192_rcnn": {
        # earlier model
        "iou_fc1_ep4__vis_fc3_ep4__lr-4": [("box_sim", "box_fc1", 4), ("OSNet_sim", "visual_osn_fc3", 4)],
        # fully trained
        "iou_fc1_ep6__vis_fc1_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc1", 6)],
        "iou_fc1_ep6__vis_fc3_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc3", 6)],
        "iou_fc1_ep6__vis_fc5_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
    },
    # pt21 gt
    "val_dl_pt21_256x192_gt": {
        # pairwise - box-visual
        "iou_fc1_ep6__vis_fc1_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc1", 6)],
        "iou_fc1_ep6__vis_fc3_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc3", 6)],
        "iou_fc1_ep6__vis_fc5_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
        # pairwise - box-pose
        "iou_fc1_ep6__pose_fc1_ep6__lr-4": [("box_sim", "box_fc1", 6), ("pose_sim_coco", "pose_coco_fc1", 6)],
        "iou_fc1_ep6__pose_fc2_ep6__lr-4": [("box_sim", "box_fc1", 6), ("pose_sim_coco", "pose_coco_fc2", 6)],
        "iou_fc1_ep6__pose_conv1o15k2fc1_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_conv1o15k2fc1", 6),
        ],
        "iou_fc1_ep6__pose_conv1o15k2fc2_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_conv1o15k2fc2", 6),
        ],
        # pairwise - pose-visual
        "pose_fc1_ep6__vis_fc5_ep6__lr-4": [("pose_sim_coco", "pose_coco_fc1", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
        "pose_fc2_ep6__vis_fc5_ep6__lr-4": [("pose_sim_coco", "pose_coco_fc2", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
        "pose_conv1o15k2fc1_ep6__vis_fc5_ep6__lr-4": [
            ("pose_sim_coco", "pose_coco_conv1o15k2fc1", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
        "pose_conv1o15k2fc2_ep6__vis_fc5_ep6__lr-4": [
            ("pose_sim_coco", "pose_coco_conv1o15k2fc2", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
        # triplet - fully trained
        "iou_fc1_ep6__oks_fc1_ep6__vis_fc5_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_fc1", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
        "iou_fc1_ep6__oks_conv1o15k2fc1_ep6__vis_fc5_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_conv1o15k2fc1", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
    },
    # pt21 rcnn
    "val_dl_pt21_256x192_rcnn": {
        # pairwise - box-visual
        "iou_fc1_ep6__vis_fc1_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc1", 6)],
        "iou_fc1_ep6__vis_fc3_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc3", 6)],
        "iou_fc1_ep6__vis_fc5_ep6__lr-4": [("box_sim", "box_fc1", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
        # pairwise - box-pose
        "iou_fc1_ep6__pose_fc1_ep6__lr-4": [("box_sim", "box_fc1", 6), ("pose_sim_coco", "pose_coco_fc1", 6)],
        "iou_fc1_ep6__pose_fc2_ep6__lr-4": [("box_sim", "box_fc1", 6), ("pose_sim_coco", "pose_coco_fc2", 6)],
        "iou_fc1_ep6__pose_conv1o15k2fc1_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_conv1o15k2fc1", 6),
        ],
        "iou_fc1_ep6__pose_conv1o15k2fc2_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_conv1o15k2fc2", 6),
        ],
        # pairwise - pose-visual
        "pose_fc1_ep6__vis_fc5_ep6__lr-4": [("pose_sim_coco", "pose_coco_fc1", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
        "pose_fc2_ep6__vis_fc5_ep6__lr-4": [("pose_sim_coco", "pose_coco_fc2", 6), ("OSNet_sim", "visual_osn_fc5", 6)],
        "pose_conv1o15k2fc1_ep6__vis_fc5_ep6__lr-4": [
            ("pose_sim_coco", "pose_coco_conv1o15k2fc1", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
        "pose_conv1o15k2fc2_ep6__vis_fc5_ep6__lr-4": [
            ("pose_sim_coco", "pose_coco_conv1o15k2fc2", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
        # triplet - fully trained
        "iou_fc1_ep6__oks_fc1_ep6__vis_fc5_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_fc1", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
        "iou_fc1_ep6__oks_conv1o15k2fc1_ep6__vis_fc5_ep6__lr-4": [
            ("box_sim", "box_fc1", 6),
            ("pose_sim_coco", "pose_coco_conv1o15k2fc1", 6),
            ("OSNet_sim", "visual_osn_fc5", 6),
        ],
    },
}


def set_up_test_dgs_module(cfg: Config, dl_key: str, dgs_mod_data: list[tuple[str, str, int]]) -> DGSEngine:
    """Given a configuration, modify it for the multi similarity case.
     Then create a :class:`.DGSEngine` and set up those similarity functions and load the weights for the alpha module.

    Where sim_mods is a list containing the params for ``S`` similarity modules.
    Each of the params tuples consists of

    - the name of the similarity module in the config file
    - the name of the alpha weight generation module
    - the epoch of the weights loaded in the alpha weight generation module
    """
    cfg["is_training"] = False
    cfg["DGSModule"]["names"] = [[sm[0]] for sm in dgs_mod_data]
    cfg["DGSModule"]["combine"] = "dac_test"
    base_lr = cfg["train"]["optimizer_kwargs"]["lr"]
    cfg["dac_test"]["alpha_modules"] = [a_name for _, a_name, _ in dgs_mod_data]

    engine = get_dgs_engine(cfg=cfg, dl_keys=(None, None, dl_key))

    for s_i, (sim_name, alpha_name, epoch) in enumerate(dgs_mod_data):
        checkpoints = glob(
            os.path.normpath(
                os.path.join(
                    "./results/own/train_single/",
                    f"./{dl_key.replace('val', 'train').replace('rcnn', 'gt')}/{sim_name}/{alpha_name}_{base_lr:.10f}/"
                    f"checkpoints/lr*_epoch{epoch:0>3}.pth",
                )
            )
        )
        assert len(checkpoints) == 1

        engine.load_combine_alpha_weights(fp=os.path.abspath(checkpoints[0]), new_id=s_i)

    close_all_layers(engine.model)

    return engine


@notify_on_completion_or_error(min_time=30)
@t.no_grad()
def test_pt21_dynamic_alpha(
    cfg: Config, dl_key: str, paths: list, comb_name: str, dgs_mod_data: list[tuple[str, str, int]]
) -> None:
    """Set the PT21 config."""
    subm_key = "submission_pt21"
    folder_path = f"./{dl_key}/{comb_name}/"

    # change config that is the same for all sub-datasets (videos) of pt21
    cfg["log_dir_suffix"] = os.path.join(folder_path, "./test_log/")
    cfg["test"]["submission"] = [subm_key]

    # get all the sub folders or files and analyze them one-by-one
    for sub_datapath in (pbar_data := tqdm(paths, desc="ds_sub_dir", leave=False)):
        pbar_data.set_postfix_str(os.path.basename(sub_datapath))
        sub_ds_name: str = sub_datapath.split("/")[-1].removesuffix(".json")

        # change config data that is different for each of the sub-datasets (videos) of pt21
        cfg[dl_key]["data_path"] = sub_datapath
        cfg["engine"]["writer_log_dir_suffix"] = f"./{sub_ds_name}/"

        # set the new path for the out file in the log_dir
        cfg[subm_key]["file"] = os.path.abspath(
            os.path.normpath(os.path.join(cfg["log_dir"], folder_path, f"./results_json/{sub_ds_name}.json"))
        )

        if os.path.exists(cfg[subm_key]["file"]):
            continue

        engine = set_up_test_dgs_module(cfg=cfg, dl_key=dl_key, dgs_mod_data=dgs_mod_data)

        engine.test()

        # terminate and reset log dir
        engine.terminate()


# @torch_memory_analysis
@notify_on_completion_or_error(min_time=30)
@t.no_grad()
def test_dance_dynamic_alpha(
    cfg: Config, dl_key: str, paths: list, comb_name: str, dgs_mod_data: list[tuple[str, str, int]]
) -> None:
    """Set the DanceTrack config."""
    orig_log_dir = cfg["log_dir"]

    subm_key = "submission_MOT"
    log_dir = os.path.join(cfg[dl_key]["dataset_path"], f"./results_{dl_key}/{comb_name}/")

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

        engine = set_up_test_dgs_module(cfg=cfg, dl_key=dl_key, dgs_mod_data=dgs_mod_data)

        engine.test()

        # terminate and reset log dir
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
    cfg["dac_train"]["alpha_modules"] = [alpha_mod_name]
    cfg["log_dir_suffix"] = f"./{dl_train_key}/{sim_name}/{alpha_mod_name}_{lr:.10f}/"

    # add option to resume training in later epochs
    last_state_path = None
    for epoch in range(
        cfg["train"].get("save_interval", 1), int(cfg["train"]["epochs"]) + 1, cfg["train"].get("save_interval", 1)
    ):
        weights = glob(os.path.join(cfg["log_dir"], cfg["log_dir_suffix"], f"./checkpoints/lr*_epoch{epoch:0>3}.pth"))
        if len(weights) > 0:
            # skip early, because every epoch was trained
            if epoch == int(cfg["train"]["epochs"]):
                return
            # set last state to
            assert len(weights) == 1
            last_state_path = weights[0]
        else:
            # results file does not exist yet - start engine mid-way
            break

    # modify config even more
    if "pt21" in dl_train_key:
        cfg["train"]["submission"] = ["submission_pt21"]
    elif "dance" in dl_train_key:
        cfg["train"]["submission"] = ["submission_MOT"]
    else:
        raise NotImplementedError(f"unknown type of dataloader, got: {dl_train_key}")

    cfg["train"]["load_image_crops"] = not any(val in sim_name for val in ["pose_", "box_"])

    # use the modified config and obtain the model used for training
    engine_train = get_dgs_engine(cfg=cfg, dl_keys=(dl_train_key, dl_eval_key, None))

    engine_train.logger.debug(
        f"Training on the ground-truth train-dataset with config: {dl_train_key} - {alpha_mod_name}"
    )

    # initialize the weights of the alpha module(s)
    init_model_params(engine_train.model.combine.alpha_model)

    # load pretrained checkpoint
    if last_state_path is not None:
        engine_train.initialize_optimizer()
        engine_train.load_model(path=last_state_path)

    # train the model(s)
    engine_train.train_model()
    engine_train.terminate()


if __name__ == "__main__":
    print(f"Cuda available: {t.cuda.is_available()}")

    config = load_config(CONFIG_FILE)

    # ############################## #
    # TRAINING & Accuracy EVALUATION #
    # ############################## #

    if TRAIN:
        print("#### START TRAINING ####")
        # for each of the dataloaders
        for DL_KEY in (pbar_dl := tqdm(DL_KEYS_TRAIN, desc="Dataloaders", leave=False)):
            DL_TRAIN_KEY, DL_EVAL_KEY, SIM_NAMES = DL_KEY

            # for every similarity or combination of similarities in this dataloader
            for SIM_NAME, alpha_modules in (pbar_key := tqdm(SIM_NAMES.items(), desc="similarities", leave=False)):
                pbar_key.set_postfix_str(SIM_NAME)

                # for every type of alpha module
                for ALPHA_MOD_NAME in (pbar_alpha_mod := tqdm(alpha_modules, desc="alpha_modules", leave=False)):
                    pbar_alpha_mod.set_postfix_str(ALPHA_MOD_NAME)

                    # set name
                    config["name"] = f"Train-Dynamic-Weights-Individually-{SIM_NAME}-{ALPHA_MOD_NAME}"

                    train_dynamic_alpha(
                        cfg=deepcopy(config),
                        dl_train_key=DL_TRAIN_KEY,
                        dl_eval_key=DL_EVAL_KEY,
                        alpha_mod_name=ALPHA_MOD_NAME,
                        sim_name=SIM_NAME,
                    )

    # ########## #
    # EVALUATION #
    # ########## #

    if EVAL:
        print("#### START EVALUATION ####")
        for DL_KEY, similarities in (pbar_dl := tqdm(DL_KEYS_EVAL.items(), desc="datasets", leave=False)):
            pbar_dl.set_postfix_str(DL_KEY)

            for COMB_NAME, DGS_MODULE_DATA in (
                pbar_key := tqdm(similarities.items(), desc="similarities", leave=False)
            ):
                pbar_key.set_postfix_str(COMB_NAME)

                # TODO accept "List_" datasets and test them individually

                if "pt21" in DL_KEY:
                    test_pt21_dynamic_alpha(
                        cfg=deepcopy(config),
                        dl_key=DL_KEY,
                        paths=[os.path.normpath(f) for f in glob(config[DL_KEY]["paths"])],
                        comb_name=COMB_NAME,
                        dgs_mod_data=DGS_MODULE_DATA,
                    )
                elif "dance" in DL_KEY:
                    test_dance_dynamic_alpha(
                        cfg=deepcopy(config),
                        dl_key=DL_KEY,
                        paths=[os.path.normpath(f) for f in glob(config[DL_KEY]["paths"])],
                        comb_name=COMB_NAME,
                        dgs_mod_data=DGS_MODULE_DATA,
                    )
                else:
                    raise NotImplementedError(f"unknown type of dataloader, got: {DL_KEY}")

    # ####### #
    # TESTING #
    # ####### #

    if TEST:
        print("#### START TESTING ####")
        print("skipped for now")
