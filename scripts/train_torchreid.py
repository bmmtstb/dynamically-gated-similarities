"""
Continue training a pre-trained `torchreid` model on another dataset
====================================================================

#. Load a pre-trained `torchreid` model
#. Create an ImageDataManager containing multiple datasets

Notes:
    Assert cwd to be the DGS folder.

Notes:
    From `Omni-Scale Feature Learning for Person Re-Identification <https://arxiv.org/pdf/1905.00953.pdf>`_

    Batch size and weight decay are set to 64 and 5e-4 respectively.
    For training from scratch, SGD is used to train the network for 350 epochs.
    The learning rate starts from 0.065 and is decayed by 0.1 at 150, 225 and 300 epochs.
    Data augmentation includes random flip, random crop and random patch.
    For fine-tuning, we train the network with AMSGrad and initial learning rate of 0.0015 for 150 epochs.
    The learning rate is decayed by 0.1 every 60 epochs.
    During the first 10 epochs, the ImageNet pre-trained base network is frozen and only the randomly
    initialised classifier is open for training.
    Images are resized to 256 × 128.
    Data augmentation includes random flip and random erasing.
"""
from datetime import date
from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from dgs.models.dataset.posetrack21 import PoseTrack21Torchreid
from torchreid.data import ImageDataManager, register_image_dataset
from torchreid.engine.image.softmax import ImageSoftmaxEngine
from torchreid.models import build_model
from torchreid.optim import build_lr_scheduler, build_optimizer

if __name__ == "__main__":
    LOG_DIR = f"./results/torchreid/pose/{date.today().strftime('YYYYMMDD')}/"
    MODEL_NAME = "osnet_x1_0"
    BATCH_TRAIN = 32
    HEIGHT = 256
    WIDTH = 256
    TRANSFORMS: Union[str, list[str]] = "random_flip"
    MARKET1501_500K: bool = False
    USE_GPU: bool = True
    DIST_METRIC: str = "cosine"  # "cosine" or "euclidean"
    EPOCHS: int = 1
    LOSS: str = "softmax"  # "softmax" or "triplet"
    PRE_TRAINED: bool = False

    # noinspection PyTypeChecker
    register_image_dataset("PoseTrack21", PoseTrack21Torchreid)

    print(f"Cuda available: {torch.cuda.is_available()}")

    data_manager = ImageDataManager(
        root="./data/",
        # sources=["market1501", "dukemtmcreid", "viper", "grid", "ilids", "PoseTrack21"],
        sources=["PoseTrack21"],
        height=HEIGHT,
        width=WIDTH,
        batch_size_train=BATCH_TRAIN,
        use_gpu=USE_GPU,
        workers=0,
        # num_cams=1,
        market1501_500k=MARKET1501_500K,
    )
    print(f"train pids: {data_manager.num_train_pids}")

    # Specify ReID model with num_classes = max(nof pids) -> in PT21
    m = build_model(
        name=MODEL_NAME, num_classes=5474, use_gpu=USE_GPU, pretrained=PRE_TRAINED, loss=LOSS
    ).cuda()  # send model to cuda, because torchreid doesn't do it properly

    writer = SummaryWriter(log_dir=LOG_DIR, filename_suffix=f"{MODEL_NAME}")

    o = build_optimizer(model=m)
    s = build_lr_scheduler(optimizer=o)
    engine = ImageSoftmaxEngine(datamanager=data_manager, model=m, optimizer=o, scheduler=s, use_gpu=USE_GPU)
    # set a custom writer in engine
    engine.writer = writer

    print("Run Engine and save model after every epoch")
    engine.run(
        save_dir=LOG_DIR,
        dist_metric=DIST_METRIC,
        max_epoch=EPOCHS,
    )
