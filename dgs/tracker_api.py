"""
Base Tracker API structure of tracking via dynamically gated similarities
"""

from dgs.models.dataset.dataset import BaseDataset
from dgs.models.embedding_generator.reid import EmbeddingGeneratorModule
from dgs.models.loader import get_data_loader, module_loader
from dgs.models.module import enable_keyboard_interrupt
from dgs.models.states import DataSample
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
        # self.m_backbone: BackboneModule = module_loader(self.cfg, "backbone")
        # self.m_backbone = torch.compile(self.m_backbone)

        self.m_vis_reid: EmbeddingGeneratorModule = module_loader(self.cfg, "visual_embedding_generator")
        # self.m_vis_siml: SimilarityModule = module_loader(self.cfg, "visual_similarity")

        # self.m_pose_reid: EmbeddingGeneratorModule = module_loader(self.cfg, "pose_embedding_generator")
        # self.m_pose_warp: PoseWarpingModule = module_loader(self.cfg, "pose_warping_module")
        # self.m_pose_siml: SimilarityModule = module_loader(self.cfg, "pose_similarity")

        # self.m_alpha = ...

    @enable_keyboard_interrupt
    def run(self) -> None:
        """Run Tracker."""
        # dataset
        dataset: BaseDataset = module_loader(self.cfg, "dataset")

        # dataloader
        data_loader = get_data_loader(self.cfg, dataset)
        for batch_idx, batch in enumerate(data_loader):
            batch: DataSample  # fixme correct? or is it list of dict, or is it dict with fancy torch stuff
            print(batch_idx)
            print(batch)

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
        model_names = ["backbone", "m_vis_reid", "m_vis_siml", "m_pose_reid", "m_pose_warp", "m_pose_siml"]
        for name in model_names:
            if (
                hasattr(self, name)  # Model exists
                and hasattr(getattr(self, name), "terminate")  # model has attribute terminate
                and callable(getattr(getattr(self, name), "terminate"))  # model.terminate is callable
            ):
                getattr(self, name).terminate()
