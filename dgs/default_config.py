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

# cfg.name = "DEFAULT"  # shouldn't be set, to force user to give it a name
cfg.print_prio = "normal"
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
cfg.dataset.path = "./posetrack_data_fast/val/"
# cfg.dataset.path = "./posetrack_data_fast/val/000342_mpii_test.json"

cfg.dataset.crop_mode = "zero-pad"
cfg.dataset.crop_size = (256, 256)  #

# ################ #
# Visual Embedding #
# ################ #
cfg.visual_embedding_generator = EasyDict()
cfg.visual_embedding_generator.module_name = "torchreid"  # module name
cfg.visual_embedding_generator.model_name = "osnet_ain_x1_0"  # torchreid model name (if applicable)
cfg.visual_embedding_generator.embedding_size = 128
cfg.visual_embedding_generator.weights = (
    "./weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)

cfg.visual_similarity = EasyDict()
cfg.visual_similarity.module_name = "cosine"

# ############## #
# Pose Embedding #
# ############## #
cfg.pose_embedding_generator = EasyDict()
cfg.pose_embedding_generator.module_name = "LinearPBEG"
cfg.pose_embedding_generator.embedding_size = 16
cfg.pose_embedding_generator.hidden_layers = []
cfg.pose_embedding_generator.joint_shape = (17, 2)  # (J, kp_dim)
cfg.pose_embedding_generator.bbox_format = "XYWH"  #
# cfg.pose_embedding_generator.weights = "./weights/dummy.pth"

cfg.pose_similarity = EasyDict()
cfg.pose_similarity.module_name = "euclidean"


# ############# #
# Warping Model #
# ############# #
cfg.pose_warping_module = EasyDict()
cfg.pose_warping_module.module_name = "kalman"
cfg.pose_warping_module.weights = ""


# ######################## #
# Combine the Similarities #
# ######################## #
cfg.combined_similarity = EasyDict()
cfg.combined_similarity.module_name = "static_alpha"
cfg.combined_similarity.alpha = [0.6, 0.4]
