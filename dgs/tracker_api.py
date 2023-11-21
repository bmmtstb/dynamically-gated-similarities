"""
Base Tracker API structure of tracking via dynamically gated similarities
"""

from dgs.models.backbone.backbone import BackboneModule
from dgs.models.embedding_generator.reid import EmbeddingGeneratorModule
from dgs.models.loader import module_loader
from dgs.utils.config import fill_in_defaults, load_config
from dgs.utils.types import FilePath


class DGSTracker:
    """Basic API for the dynamically gated similarities tracker

    Most parameters can be customized using the tracker configuration file.
    If a parameter isn't given, this tracker will use the default values from dgs.default_config.

    This tracker has a modular structure, and most modules are replaceable by a number of other modules.
    Additionally, if a module is missing, new modules are straight-forward to implement by inheriting from the
    different classes of base modules.

    If you want a tracker with a different structure,
        feel free to create your own custom DGSTracker class that uses the different modules.

    Examples:
        You can either run the file `./scripts.demo_track.py` or in the root directory run:

        .. code-block:: python

            from dgs.tracker_api import DGSTracker

            DGSTracker(tracker_cfg="./configs/simplest_tracker.yaml")
    """

    # this API will have many references to other models and therefore will have many attributes
    # pylint: disable=too-many-instance-attributes

    def __init__(self, tracker_cfg: FilePath) -> None:
        """
        Initialize the tracker by loading its configuration and setting up all of its modules.

        Args:
            tracker_cfg: Name of the configuration file either as string with its path or as python path.
        """
        # load config and possibly add some default values without overwriting custom values
        self.cfg = fill_in_defaults(load_config(tracker_cfg))

        # set up models
        self.backbone: BackboneModule = module_loader(self.cfg, "backbone")

        self.m_vis_reid: EmbeddingGeneratorModule = module_loader(self.cfg, "visual_embedding_generator")
        # self.m_vis_siml: SimilarityModule = module_loader(self.cfg, "visual_similarity")

        # self.m_pose_reid: EmbeddingGeneratorModule = module_loader(self.cfg, "pose_embedding_generator")
        # self.m_pose_warp: PoseWarpingModule = module_loader(self.cfg, "pose_warping_module")
        # self.m_vis_siml: SimilarityModule = module_loader(self.cfg, "pose_similarity")

        self.m_alpha = ...
        self.ltm = ...
        self.wm = ...

    def update(self):
        """
        One step of tracking algorithm.
        Either predict new state or load new results
        """
        raise NotImplementedError

    def load(self) -> None:
        """Load all weights for tracker, either given config or dict of weight paths"""
