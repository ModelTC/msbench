import torch.nn as nn
from typing import Dict, Callable, Any
import msbench.nn.intrinsic.pruner as msbnn

SUPPORT_CONV = (msbnn.Conv2d)
SUPPORT_LINEAR = (msbnn.Linear)

DEFAULT_REFERENCE_STATIC_SPARSE_MODULE_MAPPINGS: Dict[Callable, Any] = {
    nn.Linear: msbnn.Linear,
    nn.Conv2d: msbnn.Conv2d
}
