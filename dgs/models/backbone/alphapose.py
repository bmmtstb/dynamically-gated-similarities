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
from dgs.utils.types import Config, NodePath

ap_default_args: Config = EasyDict(
    {
        "qsize": 1024,  # length of result buffer, reduce for less CPU load
        "gpus": "0",
        "flip": False,  # enable flip testing
    }
)


class AlphaPoseBackbone(BackboneModule):
    """Use AlphaPose as Backbone to get the current state"""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        # create custom ap cfg
        self.ap_cfg = self._init_ap_args()
        # initialize detection loader
        self._init_loader()
        # start detection loader
        self.det_worker = self.det_loader.start()

    def _init_ap_args(self) -> Config:
        """
        AlphaPose needs a custom dict for args and cfg
        """
        # load ap config file using given path
        ap_cfg_file: EasyDict = load_config(self.params.cfg_path)
        # load or set default params / args for values that AlphaPose needs
        args: Config = EasyDict(
            {
                "detector": self.params.get("detector", ap_cfg_file["DETECTOR"]["NAME"]),
            }
        )
        return args

    def _init_loader(self) -> None:
        """
        Initialize file detection loader
        Either Webcam, from existing detections or live detection from images
        """

        if self.params.mode == "detfile":  # load already existing detections
            if not os.path.isfile(self.params.data):
                raise InvalidParameterException(
                    "Backbone - AlphaPose: in detfile mode, data must refer to a detection json file, not a directory."
                )

            detfile = self.params.data

            self.det_loader = FileDetectionLoader(
                input_source=detfile,
                cfg=...,
                opt=...,
                queueSize=...,
            )
            self.det_worker = self.det_loader.start()
        elif self.params.mode == "webcam":  # stream input from webcam
            # set detection batch size per GPU to 1
            self.params.detbatch = 1

            self.det_loader = WebCamDetectionLoader(
                input_source=int(self.params.webcam),
                detector=get_ap_detector(),
                cfg=...,
                opt=...,
                queueSize=...,
            )
        elif self.params.mode == "video":  # local video file
            if not os.path.isfile(self.params.data):
                raise InvalidParameterException(
                    "Backbone - AlphaPose: in video mode, data must refer to a single video file, not a directory."
                )

            videofile = self.params.data

            self.det_loader = DetectionLoader(
                input_source=videofile,
                detector=...,
                cfg=...,
                opt=...,
                mode=...,
                batchSize=...,
                queueSize=...,
            )

        elif self.params.mode == "image":  # local image(s) in path or folder-structure
            if os.path.isfile(self.params.data):  # single image
                ...
            elif os.path.isdir(self.params.data):  # single folder
                ...
            elif isinstance(self.params.data, (list, tuple)):  # iterable of image names
                if any(not os.path.isfile(fp) for fp in self.params.data):
                    raise InvalidParameterException(
                        "Backbone AlphaPose: in image list mode, one or multiple images of data do not exist."
                    )
                # ...
            else:
                raise InvalidParameterException(
                    f"Backbone AlphaPose: in image mode, could not retrieve image(s) with data {self.params.data}."
                )
            self.det_loader = DetectionLoader(
                input_source=...,
                detector=...,
                cfg=...,
                opt=...,
                mode=...,
                batchSize=...,
                queueSize=...,
            )

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        pass

    def precompute_values(self) -> None:
        pass

    def load_precomputed(self) -> ...:
        pass

    def forward(self, *args, **kwargs) -> ...:
        pass
