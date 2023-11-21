"""
Given config, load modules
"""
from dgs.utils.config import get_sub_config
from dgs.utils.types import Config, NodePath

module_paths: dict[str, NodePath] = {  # fixme: kind of useless, can this be removed?
    "backbone": ["backbone"],
    "visual_embedding_generator": ["visual_embedding_generator"],
    "visual_similarity": ["visual_embedding_generator", "similarity"],
    "pose_embedding_generator": ["pose_embedding_generator"],
    "pose_similarity": ["pose_embedding_generator", "similarity"],
    "pose_warping_module": ["pose_warping_module"],
}

submodules: dict[str, list[str]] = {
    "backbone": ["AlphaPose"],
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
            from dgs.models.backbone import AlphaPoseBackbone

            return AlphaPoseBackbone(config, path)
    elif module == "visual_embedding_generator":
        if model_name == "torchreid":
            from dgs.models.embedding_generator import TorchreidModel

            return TorchreidModel(config, path)
    elif module == "pose_warping_module":
        if model_name == "kalman":
            from dgs.models.pose_warping import KalmanFilterWarpingModel

            return KalmanFilterWarpingModel(config, path)

    raise NotImplementedError(f"Something went wrong while loading module '{module}'")
