import os
import random 
import torch 
import numpy as np
from enum import Enum

def _init(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Modes(Enum):
    ## TODO: 
    ... 

def  setup_all(seed=123):
    _init(seed)
    if torch.cuda.is_available():
        _setup_for_single_gpu()
    if torch.backends.mps.is_available():
        _setup_for_single_mps()
    else:
        _setup_for_single_cpu()


def _setup_for_single_cpu():
    os.environ["TORCH_DEVICE"] = "cpu"

def _setup_for_single_mps():
     os.environ["TORCH_DEVICE"] = "gpu"

def _setup_for_single_gpu():
     os.environ["TORCH_DEVICE"] = "gpu"
