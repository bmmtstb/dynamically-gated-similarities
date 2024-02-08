"""
Definition of default configuration for dynamically gated similarity tracker.

These values are used, iff the given config does not set own values.
"""

import torch
from easydict import EasyDict

cfg = EasyDict()

# ####### #
# General #
# ####### #

cfg.name = "DEFAULT"
cfg.description = "Default configuration..."
cfg.print_prio = "INFO"
cfg.working_memory_size = 30

# torch and device settings
cfg.device = "cpu"  # tests need cpu as the default device!
cfg.gpus = [0] if torch.cuda.is_available() else [-1]  # use gpu=0 or none, on multi-GPU systems use list of int
cfg.num_workers = 0  # number of subprocesses to use for data loading during torch DataLoader
cfg.sp = True  # single or multiprocess
cfg.is_training = False

# ######## #
# Training #
# ######## #
cfg.train = EasyDict()
cfg.train.epochs = 1
cfg.train.loss = "NLLLoss"
cfg.train.metric = "dummy"
cfg.train.optimizer = "Adam"
cfg.train.log_dir = "./results/"

# ####### #
# Testing #
# ####### #
cfg.test = EasyDict()
cfg.test.metric = "dummy"

# ####### #
# Dataset #
# ####### #
cfg.dataset = EasyDict()
cfg.dataset.module_name = "PoseTrack21"
cfg.dataset.dataset_path = "./data/PoseTrack21/"  # overall dataset path
# path to data (absolute, local, or within dataset)
cfg.dataset.path = "./posetrack_data/val/"
# cfg.dataset.path = "./posetrack_data/val/000342_mpii_test.json"

# ################ #
# Visual Embedding #
# ################ #
cfg.embedding_generator_visual = EasyDict()
cfg.embedding_generator_visual.module_name = "TorchreidModel"  # module name
cfg.embedding_generator_visual.model_name = "osnet_ain_x1_0"  # torchreid model name (if applicable)
cfg.embedding_generator_visual.embedding_size = 512
cfg.embedding_generator_visual.weights = (
    "./weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)

cfg.similarity_visual = EasyDict()
cfg.similarity_visual.module_name = "cosine"

# ############## #
# Pose Embedding #
# ############## #
cfg.embedding_generator_pose = EasyDict()
cfg.embedding_generator_pose.module_name = "LinearPBEG"
cfg.embedding_generator_pose.embedding_size = 16
cfg.embedding_generator_pose.hidden_layers = []
cfg.embedding_generator_pose.joint_shape = (17, 2)  # (J, kp_dim)
cfg.embedding_generator_pose.bbox_format = "XYWH"  #

cfg.similarity_pose = EasyDict()
cfg.similarity_pose.module_name = "euclidean"


# ############# #
# Warping Model #
# ############# #
cfg.pose_warping_module = EasyDict()
cfg.pose_warping_module.module_name = "kalman"
cfg.pose_warping_module.weights = ""


# ######################## #
# Combine the Similarities #
# ######################## #
cfg.weighted_similarity = EasyDict()
cfg.weighted_similarity.module_name = "static_alpha"
cfg.weighted_similarity.alpha = [0.6, 0.4]
