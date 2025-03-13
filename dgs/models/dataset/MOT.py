"""Datasets and helpers for the |MOT|_ datasets."""

import configparser
import os.path
import re
from glob import glob

import torch as t
import torchvision.tv_tensors as tvte

from dgs.models.dataset.dataset import ImageDataset, ImageHistoryDataset
from dgs.utils.config import DEF_VAL
from dgs.utils.exceptions import InvalidPathException
from dgs.utils.files import mkdir_if_missing, to_abspath
from dgs.utils.state import EMPTY_STATE, State
from dgs.utils.types import Config, Device, FilePath, ImgShape, NodePath, Validations

MOT_validations: Validations = {
    # optional
    "file_separator": ["optional", str],
    "seqinfo_path": ["optional", str],
    "seqinfo_key": ["optional", str],
    "crop_key": ["optional", str],
}

MOTHistory_validations: Validations = {
    # optional
    "file_separator": ["optional", str],
    "seqinfo_path": ["optional", str],
    "seqinfo_key": ["optional", str],
    "crop_key": ["optional", str],
}


def load_seq_ini(fp: FilePath, key: str = None) -> dict[str, any]:
    """Load a ``seqinfo.ini`` file containing the information of the Sequence.

    Example ``seqinfo.ini``::

        [Sequence]
        name=MOT20-##
        imDir=img1
        frameRate=##
        seqLength=####
        imWidth=1920
        imHeight=1080
        imExt=.jpg

    Args:
        fp: The local or absolute path to the seqinfo.ini file.
        key: The key at which the data is stored in the seqinfo.ini file.
            Default ``DEF_VAL["submission"]["MOT"]["seqinfo_key"]``.
    """
    if key is None:
        key = DEF_VAL["submission"]["MOT"]["seqinfo_key"]
    if not fp.endswith(".ini"):
        raise InvalidPathException(f"Presumed seqinfo.ini file '{fp}' does not have .ini ending.")
    fp = to_abspath(fp)
    ini_data = configparser.ConfigParser()
    ini_data.optionxform = str  # make sure camelCase of the variable names stays

    ini_data.read(fp, encoding="utf-8")
    if key not in ini_data:
        raise KeyError(f"Expected key '{key}' to be in seqinfo.ini file, but got keys: '{list(ini_data.keys())}'")
    return dict(ini_data[key])


def write_seq_ini(fp: FilePath, data: dict[str, any], space_around_delimiters: bool = None, key: str = None) -> None:
    """Write the ``seqinfo.ini`` file to a given location.

    Args:
        fp: The absolute path to the file containing the sequence information.
        data: The data to be written into the sequence file.
        space_around_delimiters: Whether to put spaces around the delimiters,
            see :func:`configparser.ConfigParser().write` for more details.
            Default ``DEF_VAL.dataset.MOT.space_around_delimiters``.
        key: The key at which the data should be stored in the seqinfo.ini file.
            Default ``DEF_VAL["submission"]["MOT"]["seqinfo_key"]``.
    """
    for value in ["name", "imDir", "frameRate", "seqLength", "imWidth", "imHeight", "imExt"]:
        if value not in data:
            raise ValueError(f"Expected '{value}' to be in data, but got '{data}'.")

    if space_around_delimiters is None:
        space_around_delimiters = DEF_VAL["dataset"]["MOT"]["space_around_delimiters"]

    if key is None:
        key = DEF_VAL["submission"]["MOT"]["seqinfo_key"]

    config = configparser.ConfigParser()
    config.optionxform = str  # make sure camelCase of the variable names stays
    # get current state
    config.read(fp, encoding="utf-8")
    # add a new key or modify the existing one
    config[key] = data
    with open(fp, "w", encoding="utf-8") as file:
        config.write(fp=file, space_around_delimiters=space_around_delimiters)


def load_MOT_file(
    fp: FilePath,
    *,
    sep: str = r",\s?",
    device: Device = "cpu",
    seqinfo_fp: FilePath = None,
    seqinfo_key: str = None,
    crop_key: str = None,
) -> tuple[list[State], dict[str, any]]:
    """Given the path to a file in the MOT format, get a list of states.
    Each State contains the data of one image and the respective detections.

    The MOT-files contain one annotation per line, each consisting of the values for:
    ``<frame>, <person_id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>``

    Notes:
        The world coordinates x,y,z are ignored for the 2D challenge and can be filled with -1 or 1.
        Similarly, the bounding boxes are ignored for the 3D challenge.
        However, each line is still required to contain exactly 10 values.

    Notes:
        It seems that the value for <conf> is not always present.

    Notes:
        All frame numbers, target IDs and bounding boxes are 1-based.

    Args:
        fp: The local or absolute path to the file containing the ground-truth information.
        sep: The separator used between every value in every line.
            The separator can contain regex expressions.
        device: The device the tensors of the State are on.
            Default "cpu".
        seqinfo_fp: The local or absolute path to the folder containing the seqinfo file for this dataset.
            With the default ``None``, and with ``fp=".../MOT20-XX/gt/gt.txt"``,
            the file is expected to be in ``.../MOT20-XX/seqinfo.ini``.
        seqinfo_key: The key to use in the seqinfo file.
            Default ``DEF_VAL["submission"]["MOT"]["seqinfo_key"]``.
        crop_key: The key to use in the seqinfo file when loading the image crop paths.
            Default ``DEF_VAL["submission"]["MOT"]["crop_key"]``.

    Raises:
        InvalidPathException if the file ending is not correct.

    Returns:
        A list containing one :class:`.State` per image, each State containing the respective annotations of the image.
    """
    # pylint: disable=too-many-arguments,too-many-locals

    if not fp.endswith((".txt", ".csv")):
        raise InvalidPathException(f"Presumed .txt file {fp} does not have .txt ending.")

    with open(fp, mode="r", encoding="utf-8") as file:
        lines = [[float(val) if "." in val else int(val) for val in re.split(sep, line.strip())] for line in file]

    dataset_path = os.path.dirname(os.path.dirname(fp))
    seqinfo_fp = seqinfo_fp if seqinfo_fp is not None else os.path.join(dataset_path, "./seqinfo.ini")
    seqinfo_key = seqinfo_key if seqinfo_key is not None else DEF_VAL["submission"]["MOT"]["seqinfo_key"]
    seqinfo: dict[str, any] = load_seq_ini(fp=seqinfo_fp, key=seqinfo_key)

    ds_id: str = re.findall(r"\d+", seqinfo["name"])[-1]

    crop_key = crop_key if crop_key is not None else DEF_VAL["dataset"]["MOT"]["crop_key"]
    crop_info: dict[str, any] = load_seq_ini(fp=seqinfo_fp, key=crop_key)

    # image and crop handling
    base_img_path: FilePath = os.path.join(dataset_path, seqinfo["imDir"])
    all_img_paths: list[FilePath] = glob(os.path.join(base_img_path, f"./*{seqinfo['imExt']}"))
    assert len(all_img_paths) == int(seqinfo["seqLength"])
    base_crop_path: FilePath = os.path.join(dataset_path, crop_info["imDir"])

    img_name_digits: int = len(os.path.basename(all_img_paths[0]).split(".")[0])
    assert all(len(os.path.basename(path).split(".")[0]) == img_name_digits for path in all_img_paths)
    img_shape: ImgShape = (int(seqinfo["imHeight"]), int(seqinfo["imWidth"]))

    # create a mapping from person id to a (custom) class id containing the dataset id
    # this is necessary, because the person ids are not unique across videos
    map_pid_to_cid: dict[int, int] = {
        int(pid): int(f"1{i}{ds_id}") for i, pid in enumerate(sorted(set(line[1] for line in lines)))
    }

    states = []
    for frame_id in range(1, int(seqinfo["seqLength"]) + 1):
        # get all annotations for the current frame id
        annos: list[list[any]] = [line for line in lines if line[0] == frame_id]
        file_paths = tuple(
            [os.path.join(base_img_path, f"{frame_id:0{img_name_digits}d}{seqinfo['imExt']}")] * max(len(annos), 1)
        )

        if len(annos) == 0:
            es = EMPTY_STATE.copy()
            es.filepath = file_paths
            es["frame_id"] = frame_id
            states.append(es)
            continue

        bboxes = tvte.BoundingBoxes(
            [anno[2:6] for anno in annos], format="XYWH", canvas_size=img_shape, dtype=t.float32, device=device
        )
        crop_paths = tuple(os.path.join(base_crop_path, f"{frame_id}_{anno[1]}{crop_info['imExt']}") for anno in annos)
        states.append(
            State(
                bbox=bboxes,
                filepath=file_paths,
                crop_path=crop_paths,
                person_id=t.tensor([anno[1] for anno in annos], device=device, dtype=t.long),
                class_id=t.tensor([map_pid_to_cid[anno[1]] for anno in annos], device=device, dtype=t.long),
                frame_id=[frame_id] * len(annos),
                validate=False,
            )
        )
    assert all("image" not in s.data for s in states)
    return states, seqinfo


def write_MOT_file(fp: FilePath, data: list[tuple[any, ...]], sep=",") -> None:  # pragma: no cover
    """Given MOT data, write it to the given path.

    Args:
        fp: The filepath to save the file to.
        data: A list containing the MOT data of every detection independently.
        sep: The separator to use between the values of every detection.
    """
    if not fp.endswith(".txt"):
        raise InvalidPathException(f"Presumed to write to a .txt file, but got '{fp}'.")
    fp = os.path.abspath(os.path.normpath(fp))
    mkdir_if_missing(os.path.dirname(fp))
    str_data = [sep.join(str(val) for val in d) + "\n" for d in data]

    with open(fp, mode="w+", encoding="utf-8") as file:
        file.writelines(str_data)


class MOTImage(ImageDataset):
    """Load a ground-truth- or detection-file in the |MOT|_ format.

    Params
    ------

    Optional Params
    ---------------

    file_separator (str, optional):
        The str or regular expression used to split the lines in the annotation file.
        Default ``DEF_VAL["dataset"]["MOT"]["file_separator"]``.
    crop_key (str, optional):
        The name of the key in the seqinfo file containing the info for the image crops.
        Default ``DEF_VAL["dataset"]["MOT"]["crop_key"]``.
    seqinfo_path (str, optional):
        The optional path to the ``seqinfo.ini`` file.
        Default ``DEF_VAL["dataset"]["MOT"]["seqinfo_path"]``.
    seqinfo_key (str, optional):
        The key to use in the seqinfo file.
        Default ``DEF_VAL["submission"]["MOT"]["seqinfo_key"]``.
    """

    data: list[State]

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(MOT_validations)

        self.data, seqinfo = load_MOT_file(
            fp=self.get_path_in_dataset(self.params["data_path"]),
            device=self.device,
            sep=self.params.get("file_separator", DEF_VAL["dataset"]["MOT"]["file_separator"]),
            crop_key=self.params.get("crop_key", DEF_VAL["dataset"]["MOT"]["crop_key"]),
            seqinfo_fp=self.params.get("seqinfo_path", DEF_VAL["dataset"]["MOT"]["seqinfo_path"]),
            seqinfo_key=self.params.get("seqinfo_key", DEF_VAL["submission"]["MOT"]["seqinfo_key"]),
        )

        if "crops_folder" not in self.params:
            seqinfo_path = self.params.get(
                "seqinfo_path", self.get_path_in_dataset(os.path.join(seqinfo["name"], "./seqinfo.ini"))
            )
            seqinfo_crop = load_seq_ini(
                fp=seqinfo_path, key=self.params.get("crop_key", DEF_VAL["dataset"]["MOT"]["crop_key"])
            )

            self.params["crops_folder"] = self.get_path_in_dataset(
                os.path.join(seqinfo["name"], str(seqinfo_crop["imDir"]))
            )
        if not os.path.exists(self.params["crops_folder"]):
            raise FileNotFoundError(f"Could not find the crops folder at '{self.params['crops_folder']}'.")

    def arbitrary_to_ds(self, a: State, idx: int) -> State:
        """Most of the state is available, now just load the image crops."""
        return a


class MOTImageHistory(ImageHistoryDataset):
    """Load a ground-truth- or detection-file in the |MOT|_ format by making sure,
    that all detections except the first ``L`` ones are loaded and are returned with the history.

    Params
    ------

    data_path (FilePath):
        The local or absolute path to the txt or csv file containing the MOT annotations.

    Optional Params
    ---------------

    file_separator (str, optional):
        The str or regular expression used to split the lines in the annotation file.
        Default ``DEF_VAL["dataset"]["MOT"]["file_separator"]``.
    crop_key (str, optional):
        The name of the key in the seqinfo file containing the info for the image crops.
        Default ``DEF_VAL["dataset"]["MOT"]["crop_key"]``.
    seqinfo_path (str, optional):
        The optional path to the ``seqinfo.ini`` file.
        Default ``DEF_VAL["dataset"]["MOT"]["seqinfo_path"]``.
    seqinfo_key (str, optional):
        The key to use in the seqinfo file.
        Default ``DEF_VAL["submission"]["MOT"]["seqinfo_key"]``.
    """

    data: list[State]

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        self.validate_params(MOTHistory_validations)

        self.data, seqinfo = load_MOT_file(
            fp=self.get_path_in_dataset(self.params["data_path"]),
            sep=self.params.get("file_separator", DEF_VAL["dataset"]["MOT"]["file_separator"]),
            device=self.device,
            crop_key=self.params.get("crop_key", DEF_VAL["dataset"]["MOT"]["crop_key"]),
            seqinfo_fp=self.params.get("seqinfo_path", DEF_VAL["dataset"]["MOT"]["seqinfo_path"]),
            seqinfo_key=self.params.get("seqinfo_key", DEF_VAL["submission"]["MOT"]["seqinfo_key"]),
        )

        if "crops_folder" not in self.params:
            seqinfo_path = self.params.get(
                "seqinfo_path", self.get_path_in_dataset(os.path.join(seqinfo["name"], "./seqinfo.ini"))
            )
            seqinfo_crop = load_seq_ini(
                fp=seqinfo_path, key=self.params.get("crop_key", DEF_VAL["dataset"]["MOT"]["crop_key"])
            )

            self.params["crops_folder"] = self.get_path_in_dataset(
                os.path.join(seqinfo["name"], str(seqinfo_crop["imDir"]))
            )
        if not os.path.exists(self.params["crops_folder"]):
            raise FileNotFoundError(f"Could not find the crops folder at '{self.params['crops_folder']}'.")

    def arbitrary_to_ds(self, a: list[State], idx: int) -> list[State]:
        return a
