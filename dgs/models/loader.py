"""
Given config, load modules
"""
from typing import Type, TypeVar

from dgs.models.backbone import get_backbone
from dgs.models.dataset import get_dataset
from dgs.models.embedding_generator import get_embedding_generator
from dgs.models.module import BaseModule
from dgs.models.pose_warping import get_pose_warping
from dgs.models.similarity import get_combined_similarity_module, get_similarity_module
from dgs.utils.config import get_sub_config
from dgs.utils.types import Config, NodePath

module_paths: dict[str, NodePath] = {  # fixme: kind of useless, can this be removed?
    "backbone": ["backbone"],
    "combined_similarity": ["combined_similarity"],
    "dataset": ["dataset"],
    "visual_embedding_generator": ["visual_embedding_generator"],
    "visual_similarity": ["visual_similarity"],
    "pose_embedding_generator": ["pose_embedding_generator"],
    "pose_similarity": ["pose_similarity"],
    "pose_warping_module": ["pose_warping_module"],
}

M = TypeVar("M", bound=BaseModule)


def module_loader(config: Config, module: str) -> M:
    """Load a given module and pass down the configuration

    Args:
        config: The configuration of the current tracker

        module: Name of submodule to load

    Returns:
        Initialized instance of the submodule with its config.
    """
    # This model will have one branch for every module
    # pylint: disable=too-many-branches

    # model config validation
    if module not in module_paths:
        raise KeyError(f"Module '{module}' has no path associated within module_paths.")

    path: NodePath = module_paths[module]
    module_name: str = get_sub_config(config, path).module_name

    m: Type[M]

    # Module import and initialization
    if module == "backbone":
        m = get_backbone(module_name)
    elif module == "combined_similarity":
        m = get_combined_similarity_module(module_name)
    elif module in ["visual_embedding_generator", "pose_embedding_generator"]:
        m = get_embedding_generator(module_name)
    elif module in ["visual_similarity", "pose_similarity"]:
        m = get_similarity_module(module_name)
    elif module == "pose_warping_module":
        m = get_pose_warping(module_name)
    elif module == "dataset":
        m = get_dataset(module_name)
    else:
        raise NotImplementedError(f"Something went wrong while loading the module '{module}'")

    # instantiate module with its configuration and path
    return m(config, path)
