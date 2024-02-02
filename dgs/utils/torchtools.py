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
from typing import Union

import torch
from torch import nn
from torch.nn import Module

from dgs.models.module import BaseModule
from dgs.utils.files import mkdir_if_missing
from dgs.utils.types import FilePath


def save_checkpoint(
    state: dict[str, any],
    save_dir: FilePath,
    is_best: bool = False,
    verbose: bool = True,
) -> None:
    r"""Save a given checkpoint.

    Args:
        state: State dictionary. See examples.
        save_dir: directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        verbose (bool, optional): whether to print a confirmation when the checkpoint has been created. Default is True.

    Examples:
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'rank1': 0.5,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)
    # save
    epoch = state["epoch"]
    fpath = os.path.join(save_dir, "model.pth.tar-" + str(epoch))
    torch.save(state, fpath)
    if verbose:
        print(f"Checkpoint saved to '{fpath}'")
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), "model-best.pth.tar"))
        if verbose:
            print("Saved best model as model-best.pth.tar")


def load_checkpoint(fpath) -> dict:
    r"""Load a given checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples:
        >>> from torchreid.utils import load_checkpoint
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


def resume_from_checkpoint(fpath: FilePath, model: nn.Module, optimizer=None, scheduler=None) -> int:
    r"""Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath (FilePath): path to checkpoint.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (LRScheduler, optional): an LRScheduler.

    Returns:
        int: start_epoch.

    Examples:
        >>> from torchreid.utils import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    print(f"Loading checkpoint from '{fpath}'")
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded model weights")
    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded optimizer")
    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded scheduler")
    start_epoch = checkpoint["epoch"]
    print(f"Last epoch = {start_epoch}")
    if "rank1" in checkpoint.keys():
        print(f"Last rank1 = {checkpoint['rank1']:.1%}")
    return start_epoch


def set_bn_to_eval(m: nn.Module) -> None:
    r"""Sets BatchNorm layers to eval mode.

    Args:
        m (nn.Module): A torch module.
    """
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def open_all_layers(model: nn.Module) -> None:
    r"""Opens all layers in this model for training.

    Args:
        model (nn.Module): A torch module.

    Examples:
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model: nn.Module, open_layers: str | list[str]) -> None:
    r"""Opens specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): A torch module.
        open_layers (str or list): layers open for training.

    Examples:
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(model, layer), f"{layer} is not an attribute of the model, please provide the correct name"

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def load_pretrained_weights(model: nn.Module, weight_path: FilePath) -> None:
    r"""Loads pretrianed weights to model.

    Features:
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing 'module.'.

    Args:
        model (nn.Module): A torch module.
        weight_path (FilePath): path to pretrained weights.

    Examples:
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
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


def configure_torch_module(
    orig_cls: Union[BaseModule, Module], name: str | None = None
) -> BaseModule:  # pragma: no cover
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

    def class_wrapper(self: Union[BaseModule, Module], *args, **kwargs):
        if not isinstance(self, BaseModule) or not isinstance(self, Module):
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
