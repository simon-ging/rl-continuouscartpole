"""Save seed function (should be deterministic)"""
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import cuda


def get_device(use_cuda):
    """Return torch device

    Args:
        use_cuda (bool): Use GPU if available
    """
    return (torch.device(
        "cuda:0" if torch.cuda.is_available() and use_cuda
        else "cpu"))


# noinspection PyUnresolvedReferences
def set_seed(seed):
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
