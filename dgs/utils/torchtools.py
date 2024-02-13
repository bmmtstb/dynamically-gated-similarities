"""
Tools for handling recurring torch tasks. Mostly taken from the `torchreid package
<https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/utils/torchtools.html#load_pretrained_weights>`_
"""

import os
import pickle
import shutil
import warnings
from collections import OrderedDict
from functools import partial
from typing import TypeVar, Union

import torch
from torch import nn, optim
from torch.nn import Module as TorchModule

from dgs.models.module import BaseModule
from dgs.utils.files import mkdir_if_missing
from dgs.utils.types import FilePath

BaseMod = TypeVar("BaseMod", bound=BaseModule)
TorchMod = TypeVar("TorchMod", bound=TorchModule)


def _get_model_from_module(module: Union[TorchMod, BaseMod]) -> TorchMod:
    """Given either a torch module or an instance of BaseModule, return a torch module.
    Within a BaseModule, this function searches for a 'module' attribute.

    Args:
        module: The module containing or being a torch module.

    Returns:
        An instance of a torch module.

    Raises:
        ValueError if a torch module cannot be found
    """
    if isinstance(module, nn.DataParallel):
        module = module.module

    if isinstance(module, BaseModule):
        if hasattr(module, "model"):
            module = module.model
        elif hasattr(module, "module"):
            module = module.module
        elif not isinstance(module, nn.Module):
            raise ValueError(
                f"model {module.__class__.__name__} is a BaseModule but there is no 'model' attribute "
                f"and the model is not a subclass of nn.Module."
            )
    return module


def save_checkpoint(
    state: dict[str, any],
    save_dir: FilePath,
    is_best: bool = False,
    remove_module_from_keys: bool = False,
    verbose: bool = True,
) -> None:
    """Save a given checkpoint.

    Args:
        state: State dictionary. See examples.
        save_dir: directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys: Whether to remove the 'module.' prepend in the state dict of the module.
        verbose (bool, optional): whether to print a confirmation when the checkpoint has been created. Default is True.

    Examples:
        >>> state = {
        >>>     'model': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'rank1': 0.5,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)
    # all the module keys start with 'module.' remove that
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["module"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["module"] = new_state_dict

    # save
    epoch = int(state["epoch"])
    fpath = os.path.join(save_dir, f"epoch-{epoch:0>3}.pth")
    torch.save(state, fpath)
    if verbose:
        print(f"Checkpoint saved to '{fpath}'")
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), "model-best.pth.tar"))
        if verbose:
            print("Saved best model as model-best.pth.tar")


def load_checkpoint(fpath) -> dict:
    """Load a given checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples:
        >>> from dgs.utils.torchtools import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")
    fpath = os.path.abspath(os.path.expanduser(fpath))
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File is not found at '{fpath}'")
    map_location = None if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print(f"Unable to load checkpoint from '{fpath}'")
        raise
    return checkpoint


def resume_from_checkpoint(
    fpath: FilePath,
    model: Union[TorchMod, BaseMod],
    optimizer: optim.Optimizer = None,
    schedulers: list[optim.lr_scheduler.LRScheduler] = None,
    verbose: bool = False,
) -> int:
    """Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath: The path to checkpoint. Can be a local or absolute path.
        model: The model that is currently trained.
        optimizer: An Optimizer.
        schedulers: List containing one or multiple LRSchedulers.
        verbose:


    Returns:
        int: start_epoch.

    Examples:
        >>> from dgs.utils.torchtools import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    model = _get_model_from_module(module=model)

    if verbose:
        print(f"Loading checkpoint from '{fpath}'")
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["model"])
    if verbose:
        print("Loaded model weights")
    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        if verbose:
            print("Loaded optimizer")
    if schedulers is not None and "scheduler" in checkpoint.keys():
        for name, scheduler in checkpoint["schedulers"].items():
            schedulers[name].load_state_dict(scheduler)
        if verbose:
            print("Loaded schedulers")
    return checkpoint["epoch"]


def set_bn_to_eval(module: Union[TorchMod, BaseMod]) -> None:
    """Sets BatchNorm layers to eval mode.

    Args:
        module: A torch module.
    """
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    module = _get_model_from_module(module=module)
    classname = module.__class__.__name__
    if classname.find("BatchNorm") != -1:
        module.eval()


def open_specified_layers(model: Union[TorchMod, BaseMod], open_layers: str | list[str], verbose: bool = False) -> None:
    """Opens the specified layers in the given model for training while keeping all other layers frozen.

    Args:
        model: A torch module or a BaseModule containing a torch module as attribute 'module'.
        open_layers: Name or names of the layers to open for training.
        verbose: Whether to print some debugging information.

    Examples:
        In the first example open only the classifier-layer,
        in the second one update the classifier and the fc-layer.

        >>> from dgs.utils.torchtools import open_specified_layers
        >>> open_specified_layers(model, open_layers='classifier')
        >>> open_specified_layers(model, open_layers=['fc', 'classifier'])

    Raises:
        ValueError if a value in open_layers is not an attribute of the model.
    """
    model = _get_model_from_module(module=model)

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        if not hasattr(model, layer):
            raise ValueError(
                f"{layer} is not an attribute of the model {model.__class__.__name__}, "
                f"please provide the correct name or model."
            )

    nof_opened: int = 0
    sub_module: TorchMod

    for name, sub_module in model.named_children():
        if name in open_layers:
            sub_module.train()
            sub_module.requires_grad_()
            nof_opened += 1
        else:
            sub_module.eval()
            sub_module.requires_grad_(False)

    if verbose:
        print(f"Opened {nof_opened} layers and {len([model.children()]) - nof_opened} layers remain closed.")


def open_all_layers(model: Union[TorchMod, BaseMod]) -> None:
    """Opens all layers in this model for training.

    Args:
        model: A torch module.

    Examples:
        >>> from dgs.utils.torchtools import open_all_layers
        >>> open_all_layers(model)
    """

    def open_module(m: TorchMod) -> None:
        if hasattr(m, "requires_grad"):
            m.requires_grad = True
        if hasattr(m, "train"):
            m.train()

    model: TorchMod = _get_model_from_module(module=model)

    model.train()
    model.requires_grad_()
    model.apply(open_module)


def load_pretrained_weights(model: TorchMod, weight_path: FilePath) -> None:
    """Loads pretrianed weights to model.

    Features:
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing 'module.'.

    Args:
        model: A torch module.
        weight_path: path to pretrained weights.

    Examples:
        >>> from dgs.utils.torchtools import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f"The pretrained weights '{weight_path}' cannot be loaded, "
            f"please check the key names manually "
            f"(** ignored and continue **)"
        )
    else:
        print(f"Successfully loaded pretrained weights from '{weight_path}'")
        if len(discarded_layers) > 0:
            print(f"** The following layers are discarded due to unmatched keys or layer size: {discarded_layers}")


def configure_torch_module(orig_cls: Union[BaseMod, TorchMod], name: str | None = None) -> BaseMod:  # pragma: no cover
    """Decorator to decorate a class, which has to be a child of torch.nn.Module and the BaseModule!
    The decorator will then call BaseModule.configure_torch_model on themselves after initializing the original class.

    If ``name`` is `None` the whole class will be used as `torch.nn.Module` which is going to be configured.
    Otherwise, the classes `name` attribute will be used as `torch.nn.Module` for configuration.

    Args:
        orig_cls: The decorated class.
        name: The name of `orig_cls`'s attribute / property which contains the `nn.Module` that should be configured.

    Raises:
        ValueError: If the class is not a child of both required parent classes.
            Or the parameter `name` is set and does not exist in the class.

    Returns:
        The decorated class after the configuration is applied.
    """
    orig_init = orig_cls.__init__

    def class_wrapper(self: Union[BaseMod, TorchMod], *args, **kwargs):
        if not isinstance(self, BaseModule) or not isinstance(self, TorchModule):
            raise ValueError(f"Given class or function {self} is not a child of BaseModule and torch.nn.Module")
        # first initialize class
        orig_init(self, *args, **kwargs)
        # then call configure_torch_model()
        # if no name is provided, use the class as torch Module, otherwise on the attribute `name`
        if name is not None:
            if name not in self:
                raise ValueError(f"Class {self} does not contain property of name {name}")
            self.configure_torch_model(module=self[name])
        else:
            self.configure_torch_module(module=self)

    # override original init method
    orig_cls.__init__ = class_wrapper
    return orig_cls
