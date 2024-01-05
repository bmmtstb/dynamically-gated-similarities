"""
Continue training a pre-trained `torchreid` model on another dataset
====================================================================

#. Load a pre-trained `torchreid` model
#. Create an ImageDataManager containing multiple datasets

Notes
    Assert cwd to be the DGS folder.
"""
import torch

from dgs.models.dataset.posetrack import PoseTrack21Torchreid
from torchreid.data import ImageDataManager, register_image_dataset
from torchreid.engine.image.softmax import ImageSoftmaxEngine
from torchreid.models import build_model
from torchreid.optim import build_lr_scheduler, build_optimizer

if __name__ == "__main__":
    # noinspection PyTypeChecker
    register_image_dataset("PoseTrack21", PoseTrack21Torchreid)

    print(f"Cuda available: {torch.cuda.is_available()}")

    data_manager = ImageDataManager(
        root="data",
        # sources=["market1501", "grid", "PoseTrack21"],
        sources=["PoseTrack21"],
        # height=256,
        width=256,
        # batch_size_train=32,
        # use_gpu=True,
        workers=0,
        # num_cams=1,
    )
    print(f"train pids: {data_manager.num_train_pids}")

    # Specify ReID model
    model = build_model(
        name="osnet_x1_0",
        num_classes=6878,  # max person id of PT21
        # use_gpu=True,
        # pretrained=True,
        # loss="softmax",
    ).cuda()  # why? but is necessary!

    optim = build_optimizer(model=model)
    sched = build_lr_scheduler(optimizer=optim)
    engine = ImageSoftmaxEngine(
        datamanager=data_manager,
        model=model,
        optimizer=optim,
        scheduler=sched,
        # use_gpu=True,
    )

    print("Run Engine and save model after every epoch")
    engine.run(
        save_dir="./results/torchreid/",
        # dist_metric="cosine",
        max_epoch=1,
    )
