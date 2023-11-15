"""
Use the AlphaPose repository (or a custom Fork) as backbone

Original AlphaPose: https://github.com/MVIG-SJTU/AlphaPose/
My AlpaPose fork with a few QoL improvements: https://github.com/bmmtstb/AlphaPose/tree/DGS
"""
import os

from easydict import EasyDict

from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from dgs.models.backbone.backbone import BackboneModule
from dgs.utils.types import Config, NodePath


class AlphaPoseBackbone(BackboneModule):
    """Use AlphaPose as Backbone to get the current state"""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)

        # initialize detection loader
        self._init_loader()
        # start detection loader
        self.det_worker = self.det_loader.start()

        self.ap_cfg: EasyDict = update_config(config.cfg_path)

    def _init_loader(self) -> None:
        """
        Initialize file detection loader
        Either Webcam, from existing detections or live detection from images
        """

        if self.params.mode == "detfile":  # load already existing detections
            if not os.path.isfile(self.params.data):
                raise IOError("AlphaPose Error: detfile must refer to a detection json file, not a directory.")

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
                detector=...,
                cfg=...,
                opt=...,
                queueSize=...,
            )
        elif self.params.mode == "video":  # local video file
            if not os.path.isfile(self.params.data):
                raise IOError("AlphaPose Error: video must refer to a video file, not a directory.")

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
                    raise IOError("AlphaPose Error: One or multiple images do not exist.")
                # ...
            else:
                raise IOError(f"AlphaPose Error: could not retrieve image(s) with data {self.params.data}")
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
