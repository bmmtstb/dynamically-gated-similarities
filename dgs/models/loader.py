"""
Given config, load modules
"""

from typing import Type, TypeVar

from dgs.models.dataset import get_dataset
from dgs.models.dataset.posetrack21 import get_pose_track_21
from dgs.models.embedding_generator import get_embedding_generator
from dgs.models.engine import get_engine
from dgs.models.module import BaseModule
from dgs.models.pose_warping import get_pose_warping
from dgs.models.similarity import get_combined_similarity_module, get_similarity_module
from dgs.utils.config import get_sub_config
from dgs.utils.types import Config, NodePath

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
    path: NodePath = [module]
    module_name: str = get_sub_config(config, path)["module_name"]

    m: Type[M]

    # Module import and initialization
    if module == "engine":
        m = get_engine(module_name)
        path = []
    elif module == "weighted_similarity":
        m = get_combined_similarity_module(module_name)
    elif module.startswith("embedding_generator_"):
        m = get_embedding_generator(module_name)
    elif module.startswith("similarity_"):
        m = get_similarity_module(module_name)
    elif module == "pose_warping_module":
        m = get_pose_warping(module_name)
    elif module.startswith("dataset_"):
        # special case: the concatenated PT21 dataset is loaded via function not class
        if module_name == "PoseTrack21":
            return get_pose_track_21(config, path)
        m = get_dataset(module_name)
    else:
        raise NotImplementedError(f"Something went wrong while loading the module '{module}'")

    # instantiate module with its configuration and path
    return m(config, path)
