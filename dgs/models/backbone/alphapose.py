"""
Use the AlphaPose repository (or a custom Fork) as backbone

Original AlphaPose: https://github.com/MVIG-SJTU/AlphaPose/
My AlpaPose fork with a few QoL improvements: https://github.com/bmmtstb/AlphaPose/tree/DGS
"""
import os
import warnings
from typing import Union

from easydict import EasyDict
from natsort import natsorted

from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from detector.apis import get_detector as get_ap_detector  # alphapose detector
from dgs.models.backbone.backbone import BackboneModule
from dgs.utils.config import load_config
from dgs.utils.exceptions import InvalidParameterException
from dgs.utils.types import Config, NodePath, Validations
from dgs.utils.utils import is_dir, is_file, project_to_abspath

ap_default_args: Config = EasyDict(
    {
        "qsize": 1024,  # length of result buffer, reduce for less CPU load
        "gpus": "0",
        "flip": False,  # enable flip testing
        "detbatch": 5,  # batchsize of detector
        "sp": False,  # use a single cuda process
    }
)

ap_validations: Validations = {
    "mode": [("in", ["detfile", "image", "webcam", "video"])],
    "data": ["not None"],
    "cfg_path": ["str", "file exists in project", ("endswith", ".yaml")],
}


class AlphaPoseBackbone(BackboneModule):
    """Use AlphaPose as Backbone to get the current state"""

    def __init__(self, config: Config, path: NodePath):
        super().__init__(config, path)
        self.validate_params(ap_validations)

        # load ap config file using given path
        self.ap_cfg_file: EasyDict = load_config(self.params["cfg_path"])
        # create custom AP args / options
        self.ap_args: EasyDict = self._init_ap_args()
        # initialize detection loader
        self.det_loader = self._init_loader()
        # start detection loader
        self.det_worker = self.det_loader.start()

    def _init_ap_args(self) -> EasyDict:
        """
        AlphaPose needs a custom dict for args / opt (changes names...).
        Don't confuse args / opt with the loaded config file.

        Returns:
            Additional config for AlphaPose, which is expected to be an EasyDict.
        """

        # load or set default params / args for values that AlphaPose needs
        args: Config = EasyDict()
        # either set detector name or load from config
        args.detector = (self.params.get("detector", self.ap_cfg_file["DETECTOR"]["NAME"]),)
        args.debug = (self.params.get("debug", self.config["print_prio"]) in ["debug", "all"],)
        args.qsize = (self.params.get("qsize", ap_default_args["qsize"]),)
        args.gpus = (self.params.get("gpus", ap_default_args["gpus"]),)
        args.flip = (self.params.get("flip", ap_default_args["flip"]),)
        args.detbatch = (self.params.get("detbatch", ap_default_args["detbatch"]),)
        args.sp = (self.params.get("sp", ap_default_args["sp"]),)

        return args

    def _init_loader(self) -> Union[FileDetectionLoader, DetectionLoader, WebCamDetectionLoader]:
        """Initialize detection loader.

        Detection loader can either use Webcam, existing detections, or file-based detection from images or videos.

        Choose mode via self.params["mode"]:
            - detfile: use existing detection files
            - webcam: use webcam as input for live detections
            - video: specify the path to single video
            - image: either specify a path to a single image, the path to a .txt file containing image paths,
                or the name of a folder containing images

        Returns:
            The detection loader from AlphaPose.
        """

        def init_detfile() -> FileDetectionLoader:
            """Initialize AlphaPose file loader for existing detections"""

            if not is_file(self.params["data"]):
                raise InvalidParameterException(
                    "Backbone - AlphaPose: in detfile mode, data must refer to a detection json file, not a directory."
                )

            detfile = project_to_abspath(self.params["data"])

            return FileDetectionLoader(
                input_source=detfile,
                cfg=self.ap_cfg_file,
                opt=self.ap_args,
                queueSize=self.ap_args.qsize,
            )

        def init_webcam() -> WebCamDetectionLoader:
            """Initialize AlphaPose webcam detection loader"""
            # set detection batch size per GPU to 1
            self.params["detbatch"] = 1

            return WebCamDetectionLoader(
                input_source=int(self.params["webcam"]),
                detector=get_ap_detector(self.ap_args),
                cfg=self.ap_cfg_file,
                opt=self.ap_args,
                queueSize=self.ap_args.qsize,
            )

        def init_video() -> DetectionLoader:
            """Initialize AlphaPose detection loader for single video files"""
            if not is_file(self.params["data"]):
                raise InvalidParameterException(
                    f"Backbone - AlphaPose: in video mode,"
                    f"data must refer to a single video file but is {self.params['data']}"
                )

            videofile = project_to_abspath(self.params["data"])

            return DetectionLoader(
                input_source=videofile,
                detector=get_ap_detector(self.ap_args),
                cfg=self.ap_cfg_file,
                opt=self.ap_args,
                mode=self.params["mode"],
                batchSize=self.ap_args.detbatch,
                queueSize=self.ap_args.qsize,
            )

        def init_images() -> DetectionLoader:
            """Initialize AlphaPose detection loader for image files

            Either single image file, name of txt-file containing file-paths, or whole image folder
            """
            filenames: list[str] = []
            data: str = project_to_abspath(str(self.params["data"]))

            if is_file(data) and not str(data).endswith(".txt"):  # single image file
                filenames = [data]
            elif is_dir(data):  # single folder
                for _, _, files in os.walk(data):
                    filenames = files
            elif data.endswith(".txt"):  # filename of txt file containing image names
                with open(data, encoding="utf-8") as names_file:
                    filenames = names_file.readlines()

                if any(not is_file(fp) for fp in filenames):
                    raise InvalidParameterException(
                        "Backbone AlphaPose: in list of filenames mode, one or multiple images of data do not exist."
                    )
            else:
                raise InvalidParameterException(
                    f"Backbone AlphaPose: in image mode, could not retrieve image(s) with data {self.params['data']}."
                )
            if len(filenames) == 0:
                raise InvalidParameterException(
                    f"Backbone AlphaPose: filenames is empty, but is expected to have at least one file."
                    f"mode: {self.params['mode']}, data: {self.params['data']}"
                )
            if len(filenames) == 1:
                if self.print("normal"):
                    warnings.warn("Tracking on a single image does not make sense... But this will keep going.")

            filenames = natsorted(filenames)
            return DetectionLoader(
                input_source=filenames,
                detector=get_ap_detector(self.ap_args),
                cfg=self.ap_cfg_file,
                opt=self.ap_args,
                mode=self.params["mode"],
                batchSize=self.ap_args.detbatch,
                queueSize=self.ap_args.qsize,
            )

        if self.params["mode"] == "detfile":  # load already existing detections
            return init_detfile()

        if self.params["mode"] == "webcam":  # stream input from webcam
            return init_webcam()

        if self.params["mode"] == "video":  # local video file
            return init_video()

        if self.params["mode"] == "image":  # local image(s) in path or folder-structure
            return init_images()

        raise InvalidParameterException(f"Backbone AlphaPose: invalid mode, is {self.params['mode']}")

    def load_weights(self, weight_path: str, *args, **kwargs) -> None:
        pass

    def precompute_values(self) -> None:
        pass

    def load_precomputed(self) -> ...:
        pass

    def forward(self, *args, **kwargs) -> ...:
        pass
