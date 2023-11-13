"""
Modules for different backbone models.

This backbone model is used to obtain values for:
    - imageCrop
    - joint-heatmaps
    - bbox shape
    - joint coordinates
    - (possibly) joint confidence score
    - (possibly) joint visibility score

Every Model should have capabilities to choose between different operation-modes, which can be set through the config:
    - precompute and save the values
    - load previously saved values
    - compute the values on the fly
"""
from abc import abstractmethod

from dgs.models.model import BaseModule
from dgs.utils.types import Config, Path


class BackboneModule(BaseModule):
    """Abstract class for backbone models"""

    def __call__(self, *args, **kwargs) -> any:
        """see self.forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> ...:
        """
        Obtain the model outputs for the current iteration.

        Args:
            *args:
            **kwargs:

        Returns:

        """

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def precompute_values(self) -> None:
        """
        Use the backbone model to precompute all the necessary values

        Predicts values for all given data and saves them to local files
        """
        raise NotImplementedError

    @abstractmethod
    def load_values(self) -> ...:
        """Use the backbone model to load precomputed values"""
        raise NotImplementedError


class AlphaPoseBackbone(BackboneModule):
    """
    Use the AlphaPose repository (or a custom Fork) as backbone

    AlphaPose reference: https://github.com/MVIG-SJTU/AlphaPose/
    My AlpaPose fork with a few qol improvements: https://github.com/bmmtstb/AlphaPose/tree/DGS
    """

    def __init__(self, config: Config, path: Path):
        super().__init__(config, path)

        # initialize detection loader
        self.det_loader = ...
        self._init_loader()
        # start detection loader
        self.det_worker = self.det_loader.start()

    def _init_loader(self) -> None:
        """
        Initialize file detection loader
        Either Webcam, from existing detections or live detection from images
        """
        if self.params.mode == "detfile":
            from alphapose.utils.file_detector import FileDetectionLoader  # pylint: disable=import-outside-toplevel

            self.det_loader = FileDetectionLoader(
                input_source=...,
                cfg=...,
                opt=...,
                queueSize=...,
            )
            self.det_worker = self.det_loader.start()
        elif self.params.mode == "webcam":
            from alphapose.utils.webcam_detector import WebCamDetectionLoader  # pylint: disable=import-outside-toplevel

            self.det_loader = WebCamDetectionLoader(
                input_source=...,
                detector=...,
                cfg=...,
                opt=...,
                queueSize=...,
            )
        else:
            from alphapose.utils.detector import DetectionLoader  # pylint: disable=import-outside-toplevel

            self.det_loader = DetectionLoader(
                input_source=...,
                detector=...,
                cfg=...,
                opt=...,
                mode=...,
                batchSize=...,
                queueSize=...,
            )

    def precompute_values(self) -> None:
        pass

    def load_values(self) -> ...:
        pass

    def forward(self, *args, **kwargs) -> ...:
        pass


class AlphaPoseWebcam(BackboneModule):
    """
    The AlphaPose repository allows the usage of a live webcam demo, which we here extend as backbone for DGS.
    """

    def forward(self, *args, **kwargs) -> ...:
        pass

    def precompute_values(self) -> None:
        """Not applicable for a live webcam video"""
        raise NotImplementedError

    def load_values(self) -> ...:
        pass
