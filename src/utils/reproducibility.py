"""Reproducibility utilities for controlling all sources of randomness."""

import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> int:
    """Set all random seeds for reproducibility.

    Args:
        seed: The random seed to use.
        deterministic: If True, enable deterministic algorithms and disable
            cuDNN benchmark mode. May reduce performance on some operations.

    Returns:
        The seed that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """Worker init function for reproducible DataLoader multi-processing.

    Pass as worker_init_fn to torch.utils.data.DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
