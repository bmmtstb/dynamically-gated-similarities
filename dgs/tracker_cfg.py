"""
Definition of default configuration for dynamically gated similarity tracker

These values are used, iff the given config does not set own values
"""

from easydict import EasyDict

cfg = EasyDict()

# ####### #
# General #
# ####### #

cfg.device = "cuda"
cfg.print_prio = "normal"
cfg.working_memory_size = 30

# ############## #
# Backbone Model #
# ############## #
cfg.backbone = EasyDict()
cfg.backbone.model = "AlphaPose"
cfg.backbone.cfg_path = "./dependencies/AlphaPose_Fork/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
cfg.backbone.batchsize = 8

# ################ #
# Visual Embedding #
# ################ #
cfg.visual_embedding_generator = EasyDict()
cfg.visual_embedding_generator.model = "osnet_ain_x1_0"  # reid generator class, model name if applicable
cfg.visual_embedding_generator.weights = (
    "trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0."
    "0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)
cfg.visual_embedding_generator.similarity = EasyDict()
cfg.visual_embedding_generator.similarity.model = "dummy"

# ############## #
# Pose Embedding #
# ############## #
cfg.pose_embedding_generator = EasyDict()
cfg.pose_embedding_generator.model = "dummy"
cfg.pose_embedding_generator.weights = "trackers/weights/dummy.pth"
cfg.pose_embedding_generator.similarity = EasyDict()
cfg.pose_embedding_generator.similarity.model = "dummy"


# ############# #
# Warping Model #
# ############# #
cfg.pose_warping_module = EasyDict()
cfg.pose_warping_module.model = "kalman"
cfg.pose_warping_module.weights = ""
