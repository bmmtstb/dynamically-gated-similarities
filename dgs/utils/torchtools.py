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


def get_model_from_module(module: Union[TorchMod, BaseMod]) -> TorchMod:
    """Given either a torch module or an instance of BaseModule, return a torch module.
    Within a BaseModule, this function searches for a 'module' attribute.

    Args:
        module: The module containing or being a torch module.

    Returns:
        An instance of a torch module.

    Raises:
        ValueError if a torch module cannot be found.
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
            elif k.startswith("model."):
                k = k[6:]
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
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        elif k.startswith("module.") and k[7:] in model_dict and model_dict[k[7:]].size() == v.size():
            new_state_dict[k[7:]] = v
            matched_layers.append(k[7:])
        elif k.startswith("model.") and k[6:] in model_dict and model_dict[k[6:]].size() == v.size():
            new_state_dict[k[6:]] = v
            matched_layers.append(k[6:])
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


def resume_from_checkpoint(
    fpath: FilePath,
    model: Union[TorchMod, BaseMod],
    optimizer: optim.Optimizer = None,
    scheduler: optim.lr_scheduler.LRScheduler = None,
    verbose: bool = False,
) -> int:
    """Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath: The path to checkpoint. Can be a local or absolute path.
        model: The model that is currently trained.
        optimizer: An Optimizer.
        scheduler: A single LRScheduler.
        verbose: Whether to print additional debug information.

    Returns:
        int: start_epoch.

    Examples:
        >>> from dgs.utils.torchtools import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    model = get_model_from_module(module=model)

    if verbose:
        print(f"Loading checkpoint from '{fpath}'")

    load_pretrained_weights(model=model, weight_path=fpath)

    if verbose:
        print("Loaded model weights")

    checkpoint = load_checkpoint(fpath)

    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        if verbose:
            print("Loaded optimizer")
    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
        if verbose:
            print("Loaded scheduler")
    return checkpoint["epoch"]


def set_bn_to_eval(module: Union[TorchMod, BaseMod]) -> None:
    """Sets BatchNorm layers to eval mode.

    Args:
        module: A torch module.
    """
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    module = get_model_from_module(module=module)
    classname = module.__class__.__name__
    if classname.find("BatchNorm") != -1:
        module.eval()


def open_specified_layers(
    model: Union[TorchMod, BaseMod], open_layers: str | list[str], freeze_others: bool = True, verbose: bool = False
) -> None:
    """Opens the specified layers in the given model for training while keeping all other layers unchanged or frozen.

    Args:
        model: A torch module or a BaseModule containing a torch module as attribute 'module'.
        open_layers: Name or names of the layers to open for training.
        freeze_others: Whether to freeze all the other modules that are not present in ``open_layers``.
        verbose: Whether to print some debugging information.

    Examples:
        In the first example, open only the classifier-layer and freeze the rest of the model.
        Then, in the second example using the same model,
        open the two fc-layers while keeping the previously opened classifier open.
        In the third one open the fc- and classifier-layers and freeze everything else.

        >>> from dgs.utils.torchtools import open_specified_layers
        >>> open_specified_layers(model, open_layers='classifier')
        >>> open_specified_layers(model, open_layers=['fc1', 'fc2'], freeze_others=False)
        >>> open_specified_layers(other_model, open_layers=['fc', 'classifier'])

    Raises:
        ValueError if a value in open_layers is not an attribute of the model.
    """
    # pylint: disable=too-many-branches
    model = get_model_from_module(module=model)

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        if not hasattr(model, layer):
            raise ValueError(
                f"{layer} is not an attribute of the model {model.__class__.__name__}, "
                f"please provide the correct name or model."
            )

    nof_opened, nof_freezed, still_open, still_closed = 0, 0, 0, 0
    sub_module: TorchMod

    for name, sub_module in model.named_children():
        if name in open_layers:
            sub_module.train()
            sub_module.requires_grad_()
            for p in sub_module.parameters():
                p.requires_grad = True
            nof_opened += 1
        elif freeze_others:
            sub_module.eval()
            sub_module.requires_grad_(False)
            for p in sub_module.parameters():
                p.requires_grad = False
            nof_freezed += 1
        elif sub_module.training:
            still_open += 1
        else:
            still_closed += 1

    if verbose:
        if freeze_others:
            print(f"Opened {nof_opened} layers. Froze {nof_freezed}.")
        else:
            print(f"Opened {nof_opened} layers. Layers still open: {still_open}. Layers still closed: {still_closed}")


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

    model: TorchMod = get_model_from_module(module=model)

    model.train()
    model.requires_grad_()
    model.apply(open_module)
    for p in model.parameters():
        p.requires_grad = True


def close_specified_layers(
    model: Union[TorchMod, BaseMod], close_layers: str | list[str], open_others: bool = False, verbose: bool = False
) -> None:
    """Close / Freeze the specified layers in the given model for training while keeping all other layers unchanged.

    Args:
        model: A torch module or a BaseModule containing a torch module as attribute 'module'.
        close_layers: Name or names of the layers to close for evaluation.
        open_others: Whether to open all layers not present in ``close_layers``.
        verbose: Whether to print some debugging information.

    Raises:
        ValueError if a value in close_layers is not an attribute of the model.
    """
    # pylint: disable=too-many-branches
    model = get_model_from_module(module=model)

    if isinstance(close_layers, str):
        close_layers = [close_layers]

    for layer in close_layers:
        if not hasattr(model, layer):
            raise ValueError(
                f"{layer} is not an attribute of the model {model.__class__.__name__}, "
                f"please provide the correct name or model."
            )

    nof_closed, nof_opened, still_closed, still_open = 0, 0, 0, 0
    sub_module: TorchMod

    for name, sub_module in model.named_children():
        if name in close_layers:
            sub_module.eval()
            sub_module.requires_grad_(False)
            for p in sub_module.parameters():
                p.requires_grad = False
            nof_closed += 1
        elif open_others:
            sub_module.train()
            sub_module.requires_grad_()
            for p in sub_module.parameters():
                p.requires_grad = True
            nof_opened += 1
        elif sub_module.training:
            still_open += 1
        else:
            still_closed += 1

    if verbose:
        if open_others:
            print(f"Closed {nof_closed} layers. Opened {nof_opened} layers.")
        else:
            print(f"Closed {nof_closed} layers. Still open: {still_open}, kept closed: {still_closed}")


def close_all_layers(model: Union[TorchMod, BaseMod]) -> None:
    """Closes / Freezes all layers in this model, e.g., for evaluation.

    Args:
        model: A torch module.
    """

    def freeze_module(m: TorchMod) -> None:
        if hasattr(m, "requires_grad"):
            m.requires_grad = False
        if hasattr(m, "eval"):
            m.eval()

    model: TorchMod = get_model_from_module(module=model)

    model.eval()
    model.requires_grad_(False)
    model.apply(freeze_module)
    for p in model.parameters():
        p.requires_grad = False


def configure_torch_module(orig_cls: Union[BaseMod, TorchMod], *_args, **orig_kwargs) -> BaseMod:  # pragma: no cover
    """Decorator to decorate a class, which has to be a child of torch.nn.Module and the BaseModule!
    The decorator will then call BaseModule.configure_torch_model on themselves after initializing the original class.

    If ``name`` is `None` the whole class will be used as `torch.nn.Module` which is going to be configured.
    Otherwise, the classes `name` attribute will be used as `torch.nn.Module` for configuration.

    Args:
        orig_cls: The decorated class.

    Keyword Args:
        names: The name or names of `orig_cls`'s attributes which contains the `nn.Module` that should be configured.

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
        # if no names are provided, use the class as torch Module, otherwise on the attribute `names`
        if "names" not in orig_kwargs:
            self.configure_torch_module(module=self)
        else:
            names = orig_kwargs["names"]
            if isinstance(names, str):
                names = [names]
            for name in names:
                if name not in self:
                    raise ValueError(f"Class {self} does not contain property of name {name}")
                self.configure_torch_model(module=self[name])

    # override original init method
    orig_cls.__init__ = class_wrapper
    return orig_cls


def init_model_params(module: TorchMod) -> None:
    """Given a torch module, initialize the model parameters using some default weights."""
    model: TorchMod = get_model_from_module(module)

    for instance in model.modules():
        init_instance_params(instance=instance)


def init_instance_params(instance: nn.Module) -> None:
    """Given a module instance, initialize a single instance."""
    if isinstance(instance, nn.Conv2d):
        nn.init.kaiming_normal_(instance.weight, mode="fan_out", nonlinearity="relu")
        if instance.bias is not None:
            nn.init.constant_(instance.bias, 0)
    elif isinstance(instance, nn.BatchNorm2d):
        nn.init.constant_(instance.weight, 1)
        nn.init.constant_(instance.bias, 0)
    elif isinstance(instance, nn.BatchNorm1d):
        nn.init.constant_(instance.weight, 1)
        nn.init.constant_(instance.bias, 0)
    elif isinstance(instance, nn.InstanceNorm2d):
        nn.init.constant_(instance.weight, 1)
        nn.init.constant_(instance.bias, 0)
    elif isinstance(instance, nn.Linear):
        nn.init.normal_(instance.weight, 0, 0.01)
        if instance.bias is not None:
            nn.init.constant_(instance.bias, 0)
    elif isinstance(instance, nn.ConvTranspose2d):
        nn.init.normal_(instance.weight, std=0.001)
        for name, _ in instance.named_parameters():
            if name in ["bias"]:
                nn.init.constant_(instance.bias, 0)


def torch_memory_analysis(
    f: callable, file_name: FilePath = "./memory_snapshot.pickle", max_events: int = 100_000_000
) -> callable:  # pragma: no cover
    """A decorator for torch memory analysis using :func:`torch.cuda.memory._record_memory_history`."""
    # pylint: disable=protected-access

    def decorator(*args, **kwargs):
        """The decorator."""
        try:
            # start memory recording
            torch.cuda.memory._record_memory_history(max_entries=max_events)
            # call original function
            f(*args, **kwargs)
        finally:
            torch.cuda.memory._dump_snapshot(file_name)
            # stop recording memory
            torch.cuda.memory._record_memory_history(enabled=None)

    return decorator
