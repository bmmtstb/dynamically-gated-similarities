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
import torch
from torch.utils.tensorboard import SummaryWriter

from dgs.models.dataset.posetrack import PoseTrack21Torchreid
from torchreid.data import ImageDataManager, register_image_dataset
from torchreid.engine.image.softmax import ImageSoftmaxEngine
from torchreid.models import build_model
from torchreid.optim import build_lr_scheduler, build_optimizer

if __name__ == "__main__":
    LOG_DIR = "./results/torchreid/"
    MODEL_NAME = "osnet_x1_0"
    BATCH_TRAIN = 32
    HEIGHT = 256
    WIDTH = 256
    TRANSFORMS = "random_flip"
    MARKET1501_500K = False

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
        # use_gpu=True,
        workers=0,
        # num_cams=1,
        market1501_500k=MARKET1501_500K,
    )
    print(f"train pids: {data_manager.num_train_pids}")

    # Specify ReID model
    model = build_model(
        name=MODEL_NAME,
        num_classes=5474,  # nof pids in PT21
        # use_gpu=True,
        # pretrained=True,
        # loss="softmax",
    ).cuda()  # send model to cuda, because torchreid doesn't do it properly

    writer = SummaryWriter(log_dir=LOG_DIR, filename_suffix=f"{MODEL_NAME}")

    optim = build_optimizer(model=model)
    sched = build_lr_scheduler(optimizer=optim)
    engine = ImageSoftmaxEngine(
        datamanager=data_manager,
        model=model,
        optimizer=optim,
        scheduler=sched,
        # use_gpu=True,
    )
    # set a custom writer in engine
    engine.writer = writer

    print("Run Engine and save model after every epoch")
    engine.run(
        save_dir=LOG_DIR,
        # dist_metric="cosine",
        max_epoch=1,
    )
