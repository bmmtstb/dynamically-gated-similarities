"""
Given config, load modules
"""
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from dgs.utils.config import get_sub_config
from dgs.utils.types import Config, NodePath

module_paths: dict[str, NodePath] = {  # fixme: kind of useless, can this be removed?
    "backbone": ["backbone"],
    "data": ["data"],
    "visual_embedding_generator": ["visual_embedding_generator"],
    "visual_similarity": ["visual_embedding_generator", "similarity"],
    "pose_embedding_generator": ["pose_embedding_generator"],
    "pose_similarity": ["pose_embedding_generator", "similarity"],
    "pose_warping_module": ["pose_warping_module"],
}

submodules: dict[str, list[str]] = {
    "data": ["AlphaPoseLoader"],
    "backbone": ["AlphaPose", "AlphaPoseLoader"],
    "visual_embedding_generator": ["torchreid"],
    "pose_embedding_generator": [],
    "pose_warping_module": ["kalman"],
}


def module_loader(config: Config, module: str):
    """Load a given module and pass down the configuration

    Args:
        config: The configuration of the current tracker

        module: Name of submodule to load

    Returns:
        Instance of the submodule with its config
    """
    # this model will have many module imports, to load only what we need, keep the imports local
    # pylint: disable=import-outside-toplevel

    # model config validation
    if module not in module_paths:
        raise KeyError(f"Module '{module}' has no path associated within module_paths.")

    path: NodePath = module_paths[module]
    model_name: str = get_sub_config(config, path).model

    if module not in submodules:
        raise NotImplementedError(f"Module {module} is no valid submodule.")

    if model_name not in submodules[module]:
        raise NotImplementedError(
            f"The model '{model_name}' does not exist in the module '{module}'."
            "It is most likely not a valid submodule."
        )

    # Module import and initialization
    if module == "backbone":
        if model_name == "AlphaPose":
            from dgs.models.backbone import AlphaPoseFullBackbone

            return AlphaPoseFullBackbone(config, path)
    elif module == "visual_embedding_generator":
        if model_name == "torchreid":
            from dgs.models.embedding_generator import TorchreidModel

            return TorchreidModel(config, path)
    elif module == "pose_warping_module":
        if model_name == "kalman":
            from dgs.models.pose_warping import KalmanFilterWarpingModel

            return KalmanFilterWarpingModel(config, path)
    elif module == "data":
        if model_name == "AlphaPoseLoader":
            from dgs.models.backbone.alphapose import AlphaPoseLoader

            return AlphaPoseLoader(config, path)

    raise NotImplementedError(f"Something went wrong while loading module '{module}'")


def get_data_loader(config: Config, dataset: TorchDataset) -> TorchDataLoader:
    """Set up torch DataLoader with some params from config.

    Args:
        config: Overall tracker configuration
        dataset: Reference to torch Dataset

    Returns:
        A torch DataLoader for the given dataset.
    """
    data_loader = TorchDataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )
    # https://glassboxmedicine.com/2020/03/04/multi-gpu-training-in-pytorch-data-and-model-parallelism/
    # By default, num_workers is set to 0.
    # Setting num_workers to a positive integer turns on multiprocess data loading in which data will be loaded using
    # the specified number of loader worker processes.
    # (Note that this isn’t really multi-GPU, as these loader worker processes are different processes on the CPU,
    # but since it’s related to accelerating model training, I decided to put it in the same article).
    return data_loader
