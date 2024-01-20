"""
Base Tracker API structure of tracking via dynamically gated similarities
"""

from dgs.models.dataset import get_data_loader
from dgs.models.dataset.dataset import BaseDataset
from dgs.models.embedding_generator.embedding_generator import EmbeddingGeneratorModule
from dgs.models.engine import EngineModule
from dgs.models.loader import module_loader
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.similarity.combined import CombineSimilarityModule
from dgs.models.similarity.similarity import SimilarityModule
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
        See the file `./scripts.demo_track.py` for more information.
    """

    # this API will have many references to other models and therefore will have many attributes
    # pylint: disable=too-many-instance-attributes

    @enable_keyboard_interrupt
    def __init__(self, tracker_cfg: FilePath) -> None:
        """
        Initialize the tracker by loading its configuration and setting up all of its modules.

        Args:
            tracker_cfg: Name of the configuration file either as string with its path or as python path.
        """
        # load config and possibly add some default values without overwriting custom values
        self.cfg = fill_in_defaults(load_config(tracker_cfg))

        self.ltm = ...
        self.wm = ...

        # set up models
        self.m_vis_reid: EmbeddingGeneratorModule = module_loader(self.cfg, "visual_embedding_generator")
        self.m_vis_siml: SimilarityModule = module_loader(self.cfg, "visual_similarity")

        self.m_pose_reid: EmbeddingGeneratorModule = module_loader(self.cfg, "pose_embedding_generator")
        # self.m_pose_warp: PoseWarpingModule = module_loader(self.cfg, "pose_warping_module")
        self.m_pose_siml: SimilarityModule = module_loader(self.cfg, "pose_similarity")

        self.m_alpha: CombineSimilarityModule = module_loader(self.cfg, "combined_similarity")

        # datasets and dataloaders
        test_dataset: BaseDataset = module_loader(self.cfg, "dataset")
        test_dl = get_data_loader(test_dataset, self.cfg["batch_size"])

        self.engine: EngineModule = EngineModule(self.cfg, test_loader=test_dl, get_data=..., get_target=...)

    @enable_keyboard_interrupt
    def run(self) -> None:
        """Run Tracker."""
        assert hasattr(self, "engine")
        self.engine.run()

    @enable_keyboard_interrupt
    def update(self, batch) -> None:
        """
        One step of tracking algorithm.
        Either predict new state or load new results

        Args:
            batch: A single batch of data
        """

        raise NotImplementedError

    @enable_keyboard_interrupt
    def load(self) -> None:
        """Load all weights for tracker, either given config or dict of weight paths"""
        raise NotImplementedError

    def terminate(self):
        """Terminate tracker and make sure to stop all submodules and (possible) parallel threads."""
        for attr in self.__dict__:
            # self has attribute terminate and it is a callable
            if hasattr(getattr(self, attr), "terminate") and callable(getattr(getattr(self, attr), "terminate")):
                getattr(self, attr).terminate()
