"""
Use the AlphaPose repository (or a custom Fork) as backbone

Original AlphaPose: https://github.com/MVIG-SJTU/AlphaPose/
My AlpaPose fork with a few QoL improvements: https://github.com/bmmtstb/AlphaPose/tree/DGS
"""
import os

from easydict import EasyDict

from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from detector.apis import get_detector as get_ap_detector  # alphapose detector
from dgs.models.backbone.backbone import BackboneModule
from dgs.utils.config import load_config
from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.types import Config, NodePath, Validations

ap_default_args: Config = EasyDict(
    {
        "qsize": 1024,  # length of result buffer, reduce for less CPU load
        "gpus": "0",
        "flip": False,  # enable flip testing
        "detbatch": 5,  # batchsize of detector
    }
)

ap_validation: Validations = {
    "mode": [("in", ["detfile", "image", "webcam", "video"])],
    "data": ["not None"],
    "cfg_path": ["str", "file exists", ("endswith", ".yaml")],
}


class AlphaPoseBackbone(BackboneModule):
    """Use AlphaPose as Backbone to get the current state"""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        # load ap config file using given path
        self.ap_cfg_file: EasyDict = load_config(self.params.cfg_path)
        # create custom AP args / options
        self.ap_args = self._init_ap_args()
        # initialize detection loader
        self.det_loader = self._init_loader()
        # start detection loader
        self.det_worker = self.det_loader.start()

    def _init_ap_args(self) -> Config:
        """
        AlphaPose needs a custom dict for args and cfg
        """

        # load or set default params / args for values that AlphaPose needs
        args: Config = EasyDict(
            {
                "detector": self.params.get("detector", self.ap_cfg_file["DETECTOR"]["NAME"]),
                "debug": self.params.get("debug", self.config["print_prio"]),
                "qsize": self.params.get("qsize", ap_default_args["qsize"]),
                "gpus": self.params.get("gpus", ap_default_args["gpus"]),
                "flip": self.params.get("flip", ap_default_args["flip"]),
                "detbatch": self.params.get("detbatch", ap_default_args["detbatch"]),
            }
        )
        return args

    def _init_loader(self):
        """Initialize file detection loader

        Detection loader can either use Webcam, existing detections, or file-based detection from images or videos.

        Returns:
            The detection loader from AlphaPose.
        """

        if self.params.mode == "detfile":  # load already existing detections
            if not os.path.isfile(self.params.data):
                raise InvalidParameterException(
                    "Backbone - AlphaPose: in detfile mode, data must refer to a detection json file, not a directory."
                )

            detfile = self.params.data

            return FileDetectionLoader(
                input_source=detfile,
                cfg=...,
                opt=...,
                queueSize=...,
            )

        if self.params.mode == "webcam":  # stream input from webcam
            # set detection batch size per GPU to 1
            self.params.detbatch = 1

            return WebCamDetectionLoader(
                input_source=int(self.params.webcam),
                detector=get_ap_detector(),
                cfg=...,
                opt=...,
                queueSize=...,
            )

        if self.params.mode == "video":  # local video file
            if not os.path.isfile(self.params.data):
                raise InvalidParameterException(
                    "Backbone - AlphaPose: in video mode, data must refer to a single video file, not a directory."
                )

            videofile = self.params.data

            return DetectionLoader(
                input_source=videofile,
                detector=...,
                cfg=...,
                opt=...,
                mode=...,
                batchSize=...,
                queueSize=...,
            )

        if self.params.mode == "image":  # local image(s) in path or folder-structure
            if (  # filename of txt file containing image names
                isinstance(self.params.data, str)
                and os.path.isfile(self.params.data)
                and self.params.data.endswith(".txt")
            ):
                with open(self.params.data, "r", encoding="utf-8") as names_file:
                    filenames: list[str] = names_file.readlines()

            if os.path.isfile(self.params.data) and not str(self.params.data).endswith(".txt"):  # single image file
                ...
            elif os.path.isdir(self.params.data):  # single folder
                ...
            elif isinstance(self.params.data, (list, tuple)) or filenames:  # iterable of image names
                filenames = filenames if filenames else self.params.data
                if any(not os.path.isfile(fp) for fp in filenames):
                    raise InvalidParameterException(
                        "Backbone AlphaPose: in list of filenames mode, one or multiple images of data do not exist."
                    )
                # ...
            else:
                raise InvalidParameterException(
                    f"Backbone AlphaPose: in image mode, could not retrieve image(s) with data {self.params.data}."
                )
            return DetectionLoader(
                input_source=...,
                detector=...,
                cfg=...,
                opt=...,
                mode=...,
                batchSize=...,
                queueSize=...,
            )

        raise InvalidParameterException(f"Backbone AlphaPose: invalid mode, is {self.params.mode}")

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        pass

    def precompute_values(self) -> None:
        pass

    def load_precomputed(self) -> ...:
        pass

    def forward(self, *args, **kwargs) -> ...:
        pass
