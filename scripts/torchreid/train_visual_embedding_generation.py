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

import torch
from torch.utils.tensorboard import SummaryWriter

from dgs.models.dataset.posetrack21 import PoseTrack21Torchreid
from dgs.utils.config import load_config

try:
    # If torchreid is installed using `./dependencies/torchreid`
    # noinspection PyUnresolvedReferences
    from torchreid.data import ImageDataManager, register_image_dataset

    # noinspection PyUnresolvedReferences
    from torchreid.engine import ImageSoftmaxEngine

    # noinspection PyUnresolvedReferences
    from torchreid.models import build_model

    # noinspection PyUnresolvedReferences
    from torchreid.optim import build_lr_scheduler, build_optimizer
except ModuleNotFoundError:
    # if torchreid is installed using `pip install torchreid`
    # noinspection PyUnresolvedReferences
    from torchreid.reid.data import ImageDataManager, register_image_dataset

    # noinspection PyUnresolvedReferences
    from torchreid.reid.engine import ImageSoftmaxEngine

    # noinspection PyUnresolvedReferences
    from torchreid.reid.models import build_model

    # noinspection PyUnresolvedReferences
    from torchreid.reid.optim import build_lr_scheduler, build_optimizer

MODEL_NAME = "osnet_x1_0"
BATCH_TRAIN = 128
HEIGHT = 256
WIDTH = 256
MARKET1501_500K: bool = False
USE_GPU: bool = True
DIST_METRIC: str = "euclidean"  # "cosine" or "euclidean"
EPOCHS: int = 5
LOSS: str = "softmax"  # "softmax" or "triplet"
PRE_TRAINED: bool = True
TEST_ONLY: bool = False
NUM_CLASSES: int = 5474

LOG_DIR = f"./results/torchreid/visual/{MODEL_NAME}/{date.today().strftime('%Y%m%d')}/"

CONFIG_FILE: str = "./scripts/torchreid/custom_configs/pt21_osnet_x1_0_softmax_256x192_amsgrad.yaml"

if __name__ == "__main__":

    cfg = load_config(CONFIG_FILE)

    print(f"Cuda available: {torch.cuda.is_available()}")

    # noinspection PyTypeChecker
    register_image_dataset("PoseTrack21", PoseTrack21Torchreid)

    # with HidePrint():  # this will hide the data summary and transforms set-up
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
        name=MODEL_NAME, num_classes=NUM_CLASSES, use_gpu=USE_GPU, pretrained=PRE_TRAINED, loss=LOSS
    ).cuda()  # send model to cuda, because torchreid doesn't do it properly

    writer = SummaryWriter(log_dir=LOG_DIR, filename_suffix=f"{MODEL_NAME}")

    o = build_optimizer(model=m)
    s = build_lr_scheduler(optimizer=o)
    engine = ImageSoftmaxEngine(datamanager=data_manager, model=m, optimizer=o, scheduler=s, use_gpu=USE_GPU)
    # set a custom writer in engine
    engine.writer = writer

    if TEST_ONLY:
        print("Test visual torchreid")
        engine.test(dist_metric=DIST_METRIC, save_dir=LOG_DIR)
    else:
        print("Run Engine and save model after every epoch")
        engine.run(
            save_dir=LOG_DIR,
            dist_metric=DIST_METRIC,
            max_epoch=EPOCHS,
            open_layers=["classifier"],
            eval_freq=1,
        )
