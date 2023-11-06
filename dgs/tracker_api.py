"""
overall structure of dynamically gated similarities

Tracker instance gets set up in demo_interference
    tracker = Tracker(tcfg, args)
Then track() from __init__ calls the tracker and passes the arguments
    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
"""
from dgs.models.history_warping import HistoryWarpingModel
from dgs.models.reid import EmbeddingGeneratorModel


class DGSTracker:
    """
    API for the dynamically gated similarities tracker
    """

    # this API will have many references to other models and therefore will have many attributes
    # pylint: disable=too-many-instance-attributes

    def __init__(self, tracker_cfg, program_args) -> None:
        super().__init__(tracker_cfg, program_args)

        self.cfg = tracker_cfg
        self.args = program_args
        self.num_joints = 17
        self.frame_rate = tracker_cfg.frame_rate

        self.ltm = ...
        self.wm = ...

        # set up models
        self.m_vis_reid: EmbeddingGeneratorModel = ...
        self.m_pose_reid: EmbeddingGeneratorModel = ...
        self.m_alpha = ...
        self.m_hist_warp: HistoryWarpingModel = ...

    def update(self):
        """
        one step of tracking algorithm.
        Either predict new state or load new results
        """
        raise NotImplementedError

    def load(self) -> None:
        """Load all weights for tracker, either given config or dict of weight paths"""
