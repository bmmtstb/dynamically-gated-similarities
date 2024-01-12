"""
Default Datasets for pose-based data.

PoseDataset and PoseDataManager are custom models for torchreid.
"""
from typing import Callable, Type, Union

import torch
import torchvision.transforms.v2 as tvt
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from dgs.utils.types import FilePath
from dgs.utils.utils import HidePrint
from torchreid.data import Dataset
from torchreid.data.datamanager import DataManager
from torchreid.data.sampler import build_train_sampler


class PoseDataset(Dataset):
    """Custom torchreid Dataset for pose-based data."""

    def __getitem__(self, index: int) -> dict[str, any]:
        pose_path, pid, camid, dsetid = self.data[index]
        pose = torch.load(pose_path)
        return {"img": pose, "pid": pid, "camid": camid, "dsetid": dsetid}

    def show_summary(self) -> None:
        """Print dataset summary."""
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print(f" => Loaded {self.__class__.__name__}")
        print(" ----------------------------------------")
        print("   subset   | # ids | # poses | # cameras")
        print(f"  train    | {num_train_pids:5d} | {len(self.train):8d} | {num_train_cams:9d}")
        print(f"  query    | {num_query_pids:5d} | {len(self.query):8d} | {num_query_cams:9d}")
        print(f"  gallery  | {num_gallery_pids:5d} | {len(self.gallery):8d} | {num_gallery_cams:9d}")
        print(" ----------------------------------------")


class PoseDataManager(DataManager):
    """Custom torchreid DataManager for pose-based data.

    Args:
        root (str): Root path to the directory containing all the datasets.
        sources (Type[PoseDataset] | list[Type[PoseDataset]]): The types of source pose dataset(s).
        **kwargs (dict): Additional keyword arguments, see Other Parameters below.

    Other Parameters:
        combineall (bool):
            Combine train, query and gallery in a dataset for training.
            Default is False.
        targets (Type[PoseDataset] | list[Type[PoseDataset]]):
            The types of target dataset(s).
            If not given, it equals to ``sources``.
        transforms (list[str | Callable]):
            One or multiple transformations applied to model training.
            Default is 'random_flip'.
        train_sampler (str):
            Name of the Sampler during training.
            Default "RandomSampler".
        use_gpu (bool): Whether to use the gpu.
            Default is True.
        batch_size_train (int):
            The number of images in a training batch.
            Default is 32.
        batch_size_test (int):
            The number of images in a test batch.
            Default is 32.
        num_instances (int):
            The number of instances per identity in a batch.
            Default is 4.
        num_cams (int):
            The number of cameras to sample in a batch (when using ``RandomDomainSampler``).
            Default is 1.
        num_datasets (int):
            The number of datasets to sample in a batch (when using ``RandomDatasetSampler``).
            Default is 1.
        train_sampler (str):
            Name of the torchreid sampler.
            Default is RandomSampler.
        train_sampler_t (str):
            Name of the torchreid sampler for the target train loader.
            Default is RandomSampler.
        verbose (bool):
            Print more debug information.
            Default is False.
        workers (int):
            Number of workers for the torch DataLoader.
            As long as no multi-GPU context is available, this value should not be changed.
            Default 0.

    Attributes:
        data_type (str): Is used within torchreid.
        default_kwargs: A dict of default keyword arguments.
            This dictionary is used to set default kwargs without passing hundreds of Arguments to `__init__()`.
        params (dict[str, any]): Module parameters

    Notes:
        The original image-based transforms are overwritten to support key-points as input.
    """

    data_type: str = "pose"

    default_kwargs: dict[str, any] = {
        "combineall": False,
        "targets": None,
        "transforms": ["random_flip"],
        "train_sampler": "RandomSampler",
        "use_gpu": True,
        "batch_size_train": 32,
        "batch_size_test": 32,
        "num_instances": 4,
        "num_cams": 1,
        "num_datasets": 1,
        "verbose": False,
        "workers": 0,
    }

    def __init__(
        self, root: FilePath, sources: Type[PoseDataset] | list[Type[PoseDataset]], **kwargs: dict[str, any]
    ) -> None:
        # set default kwargs
        self.params: dict[str, any] = self.default_kwargs.copy()
        self.params.update(kwargs)
        self.root = root

        # block printing of transforms
        with HidePrint():
            super().__init__(sources=sources, targets=self.params["targets"], use_gpu=self.params["use_gpu"])

        # the original Dataset transforms are initialized now, but we don't want them

        self.train_set, self.train_loader = self.load_train()
        self._num_train_pids = self.train_set.num_train_pids
        self._num_train_cams = self.train_set.num_train_cams

        self.test_loader, self.test_dataset = self.load_test()

        if self.params["verbose"]:
            self.show_summary()

    def load_train(self) -> (TorchDataset, TorchDataLoader):
        """Load the train Dataset and DataLoader as torch objects."""
        print("=> Loading train (source) dataset")
        # sum(list[Dataset]) is implemented via torchreid Dataset
        # noinspection PyTypeChecker
        train_set: Union[PoseDataset, TorchDataset] = sum(
            instance(root=self.root, mode="train", transform=self.transform_tr, instance="key_points", **self.params)
            for instance in self.sources
        )
        train_loader = TorchDataLoader(
            train_set,
            sampler=build_train_sampler(
                train_set.train,
                self.params["train_sampler"],
                batch_size=self.params["batch_size_train"],
                num_instances=self.params["num_instances"],
                num_cams=self.params["num_cams"],
                num_datasets=self.params["num_datasets"],
            ),
            batch_size=self.params["batch_size_train"],
            shuffle=False,
            num_workers=self.params["workers"],  # as long as there is no multi GPU support this has to be zero
            pin_memory=self.use_gpu,
            drop_last=True,
        )
        return train_set, train_loader

    def load_test(self) -> (dict[str, dict[str, any]], dict[str, dict[str, any]]):
        """Load the test Dataset and DataLoader as torch objects."""
        print("=> Loading test (target) dataset")
        test_loader: dict[str, dict[str, any]] = {name: {"query": None, "gallery": None} for name in self.targets}
        test_dataset: dict[str, dict[str, any]] = {name: {"query": None, "gallery": None} for name in self.targets}

        for dataset in self.targets:
            # test_loader for query
            query_set: Union[PoseDataset, TorchDataset] = dataset(
                root=self.root, mode="query", transform=self.transform_te, **self.params
            )
            # build query loader
            test_loader[dataset]["query"] = TorchDataLoader(
                query_set,
                batch_size=self.params["batch_size_test"],
                shuffle=False,
                num_workers=self.params["workers"],
                pin_memory=self.use_gpu,
                drop_last=self.params.get("drop_last_test", False),
            )

            # test_loader for gallery
            gallery_set: Union[Dataset, TorchDataset] = dataset(
                root=self.root, mode="gallery", transform=self.transform_te, **self.params
            )
            # build gallery loader
            test_loader[dataset]["gallery"] = torch.utils.data.DataLoader(
                gallery_set,
                batch_size=self.params["batch_size_test"],
                shuffle=False,
                num_workers=self.params["workers"],
                pin_memory=self.use_gpu,
                drop_last=self.params.get("drop_last_test", False),
            )

            # modify test_dataset
            test_dataset[dataset]["query"] = query_set.query
            test_dataset[dataset]["gallery"] = gallery_set.gallery
        return test_loader, test_dataset

    def show_summary(self) -> None:
        """Show a summary describing the DataManager"""
        print("\n")
        print("  **************** Summary ****************")
        print(f"  source            : {self.sources}")
        print(f"  # source datasets : {len(self.sources)}")
        print(f"  # source ids      : {self.num_train_pids}")
        print(f"  # source images   : {len(self.train_set)}")
        print(f"  # source cameras  : {self.num_train_cams}")
        print(f"  target            : {self.targets}")
        print("  *****************************************")
        print("\n")

    @staticmethod
    def build_transforms(
        transforms: Union[str, list[str], callable, list[callable]] = None, **kwargs
    ) -> (tvt.Compose, tvt.Compose):
        """Build transforms for pose data. Can't use torchreid transforms.

        Possible transforms:
        --------------------

        random_flip
            Randomly flip along the horizontal or vertical axis.
        random_horizontal_flip
            Randomly flip along the horizontal axis.
        random_vertical_flip
            Randomly flip along the vertical axis.
        random_move
             Adds normally distributed noise to the key points.
        random_resize
            Randomly resizes the key points by a factor in range (0.95, 1.05)

        Args:
            transforms: List of transform names or functions which will be applied to the data during training.
                Not used for testing!
                The transforms will be inserted into a tvt.Compose in the order they are defined in this list.
                Default is None.

        Keyword Args:
            random_horizontal_flip_prob (float): Probability of flipping the coordinates horizontally. Default 0.5
            random_vertical_flip_prob (float): Probability of flipping the coordinates vertically. Default 0.5
            random_move_prob (float): Probability to use add normally distributed movement. Default 0.5
            random_resize_prob (float): Probability to randomly resize. Default 0.5
            random_flip_prob (float): Probability of using random flipping. Default 0.5
            random_flip_probs (list[float]): When a 'random_flip' is done,
                these are the probabilities of flipping horizontal and vertical.
                Default [0.8, 0.2]

        Returns:
            (tvt.Compose, tvt.Compose): One composed transform for training and testing.

        Raises:
            ValueError: If ``transforms`` is an invalid object or contains invalid transform names.
        """

        def random_move(x: torch.Tensor) -> torch.Tensor:
            """Move a torch tensor by a little bit in random directions using a normal distribution ~N(0,1)."""
            return x + torch.randn_like(x, requires_grad=True)

        def random_resize(x: torch.Tensor) -> torch.Tensor:
            """Resize the torch tensor by a little bit, up and down. Ranges from 0.95 to 1.05."""
            return x * torch.tensor([1.0]).uniform_(0.95, 1.05)

        if transforms is None:
            transforms = []
        elif isinstance(transforms, str):
            transforms = [transforms]

        if not isinstance(transforms, list):
            raise ValueError(f"Transforms must be a list of strings, but found to be {type(transforms)}")

        train_transforms = [tvt.ToTensor(), tvt.ToDtype(dtype=torch.float32)]

        for transform in transforms:
            if transform == "random_flip":
                train_transforms.append(
                    tvt.RandomApply(
                        [
                            tvt.RandomChoice(
                                [tvt.RandomHorizontalFlip(), tvt.RandomVerticalFlip()],
                                p=kwargs.get("random_flip_probs", [0.8, 0.2]),
                            )
                        ],
                        p=kwargs.get("random_flip_prob", 0.5),
                    )
                )
            elif transform == "random_horizontal_flip":
                train_transforms.append(tvt.RandomHorizontalFlip(p=kwargs.get("random_horizontal_flip_prob", 0.5)))
            elif transform == "random_vertical_flip":
                train_transforms.append(tvt.RandomVerticalFlip(p=kwargs.get("random_vertical_flip_prob", 0.5)))
            elif transform == "random_move":
                train_transforms.append(
                    tvt.RandomApply([tvt.Lambda(random_move)], p=kwargs.get("random_move_prob", 0.5))
                )
            elif transform == "random_resize":
                train_transforms.append(
                    tvt.RandomApply([tvt.Lambda(random_resize)], p=kwargs.get("random_resize_prob", 0.5))
                )
            elif callable(transform) or isinstance(transform, Callable):
                train_transforms.append(transform)
            else:
                raise ValueError(f"Unknown transform: {transform}")

        test_transforms = [tvt.ToTensor(), tvt.ToDtype(dtype=torch.float32)]

        return tvt.Compose(train_transforms), tvt.Compose(test_transforms)
