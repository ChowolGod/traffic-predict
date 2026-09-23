"""set_seed: the single place every source of randomness is fixed (M-13, plan 3.3)."""

import random

import numpy as np
import torch


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
