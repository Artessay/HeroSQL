
import os
import random

import numpy as np
import torch
from pathlib import Path


def seed_everything(seed: int):  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def find_project_root(current: Path):
    for parent in [current] + list(current.parents):
        if (parent / '.git').exists() or (parent / 'pyproject.toml').exists():
            return parent
    return current