"""
default configuration for dynamically gated similarity tracker
"""

from easydict import EasyDict as edict

cfg = edict()
# general configuration
cfg.device = "cuda"

# memory settings
cfg.wm_size = 30

# config for visual embedding generator
cfg.vis_emb_gen = ("base", "osnet_ain_x1_0")  # reid generator class, model name if applicable
cfg.vis_emb_gen_weights = ("trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0."
                           "0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")

# config for pose embedding generator
cfg.pose_emb_gen = "dummy"
cfg.pose_emb_gen_weights = "trackers/weights/dummy.pth"

# warping model and potentially weights
cfg.warp_model = "kalman"
cfg.warp_weights = "dummy"
