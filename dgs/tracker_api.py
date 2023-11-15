"""
overall structure of dynamically gated similarities

Tracker instance gets set up in demo_interference
    tracker = Tracker(tcfg, args)
Then track() from __init__ calls the tracker and passes the arguments
    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
"""
from dgs.models.backbone.backbone import BackboneModule
from dgs.models.loader import module_loader
from dgs.models.pose_warping.pose_warping import PoseWarpingModule
from dgs.models.reid.reid import EmbeddingGeneratorModule
from dgs.models.similarity.similarity import SimilarityModule


class DGSTracker:
    """
    API for the dynamically gated similarities tracker
    """

    # this API will have many references to other models and therefore will have many attributes
    # pylint: disable=too-many-instance-attributes

    def __init__(self, tracker_cfg) -> None:
        self.cfg = tracker_cfg

        # set up models
        self.backbone: BackboneModule = module_loader(tracker_cfg, "backbone")

        self.m_vis_reid: EmbeddingGeneratorModule = module_loader(tracker_cfg, "visual_embedding_generator")
        self.m_vis_siml: SimilarityModule = module_loader(tracker_cfg, "visual_similarity")

        self.m_pose_reid: EmbeddingGeneratorModule = module_loader(tracker_cfg, "pose_embedding_generator")
        self.m_pose_warp: PoseWarpingModule = module_loader(tracker_cfg, "pose_warping_module")
        self.m_vis_siml: SimilarityModule = module_loader(tracker_cfg, "pose_similarity")

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
