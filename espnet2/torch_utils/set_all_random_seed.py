import os
import random
import warnings

import numpy as np
import torch


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False