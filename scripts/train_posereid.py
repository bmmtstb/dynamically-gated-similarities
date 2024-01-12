r"""
Train a pose-based ReID-model using the torchreid workflow
==========================================================

Workflow
--------

#. use AlphaPose to create pose estimates for Market1501 and underground_reid as long as the data does not exist
#. create / load pose-based ReID generation model
#. create DataManager for pose-based data

"""
from datetime import date
from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from dgs.models.dataset.pose_dataset import PoseDataManager
from dgs.models.dataset.posetrack21 import PoseTrack21Torchreid
from torchreid.engine import ImageSoftmaxEngine
from torchreid.models import build_model
from torchreid.optim import build_lr_scheduler, build_optimizer

if __name__ == "__main__":
    LOG_DIR = f"./results/torchreid/pose/{date.today().strftime('%Y%m%d')}/"
    MODEL_NAME = "osnet_x1_0"
    BATCH_TRAIN = 32
    TRANSFORMS: Union[str, list[str]] = ["random_horizontal_flip", "random_move"]
    USE_GPU: bool = True
    DIST_METRIC: str = "cosine"  # "cosine" or "euclidean"
    EPOCHS: int = 1
    LOSS: str = "softmax"  # "softmax" or "triplet"
    PRE_TRAINED: bool = False
    TEST_ONLY: bool = True

    print(f"Cuda available: {torch.cuda.is_available()}")

    data_manager = PoseDataManager(
        root="./data/",
        sources=[PoseTrack21Torchreid],
        batch_size_train=BATCH_TRAIN,
        use_gpu=USE_GPU,
        workers=0,
    )
    print(f"train pids: {data_manager.num_train_pids}")

    # Specify ReID model with num_classes = max(nof pids) -> in PT21
    m = build_model(
        name=MODEL_NAME, num_classes=5474, use_gpu=USE_GPU, pretrained=PRE_TRAINED, loss=LOSS
    ).cuda()  # send model to cuda, because torchreid doesn't do it properly

    writer = SummaryWriter(log_dir=LOG_DIR, filename_suffix=f"{MODEL_NAME}")

    o = build_optimizer(model=m)
    s = build_lr_scheduler(optimizer=o)
    # it seems that the ImageSoftmaxEngine doesn't really care about its inputs,
    # as long as the parsed 'data' has the 'img', 'pid' and 'camid' keys, the "image" could be a pose tensor
    engine = ImageSoftmaxEngine(datamanager=data_manager, model=m, optimizer=o, scheduler=s, use_gpu=USE_GPU)
    # set a custom writer in engine to include the model name in the output files
    engine.writer = writer

    if TEST_ONLY:
        print("Test Engine")
        engine.test(dist_metric=DIST_METRIC, save_dir=LOG_DIR)
        # 2024/01/12
        # ** Results **
        # mAP: 42.5%
        # CMC curve
        # Rank-1  : 98.1%
        # Rank-5  : 98.9%
        # Rank-10 : 99.1%
        # Rank-20 : 99.2%
    else:
        print("Run Engine (train + test)")
        print("Saves the model after every epoch and creates analysis for tensorboard.")
        engine.run(
            save_dir=LOG_DIR,
            dist_metric=DIST_METRIC,
            max_epoch=EPOCHS,
        )
