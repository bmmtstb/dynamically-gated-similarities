"""
General utility functions.
"""
import numpy as np
import torch


def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Clone and convert torch tensor to numpy.

    Args:
        t: Torch tensor on arbitrary hardware.

    Returns:
        Numpy array with the same shape and type as the original tensor
    """
    # Detach creates a new tensor with the same data, so it is important to clone.
    # May not be necessary if moving from GPU to CPU, but better safe than sorry
    return t.detach().cpu().clone().numpy()
