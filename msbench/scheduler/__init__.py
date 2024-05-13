from .base_scheduler import BaseScheduler
from .amba_scheduler import AmbaLevelPruneScheduler
from .probmask_scheduler import ProbMaskPruneScheduler
from .str_scheduler import STRScheduler
from .admm_scheduler import ADMMScheduler
from .ampere_sheduler import AmpereScheduler
from typing import Dict, Any


DEFAULT_REFERENCE_SCHEDULER_MAPPINGS: Dict[Any, Any] = {
    "BaseScheduler": BaseScheduler,
    "AmbaLevelPruneScheduler": AmbaLevelPruneScheduler,
    "ProbMaskPruneScheduler": ProbMaskPruneScheduler,
    "STRScheduler": STRScheduler,
    "AmpereScheduler": AmpereScheduler,
    "ADMMScheduler": ADMMScheduler
}


def build_sparse_scheduler(sparsity_config):
    scheduler_type = sparsity_config["scheduler"].get("type", "BaseScheduler")
    if scheduler_type == 'AmpereScheduler':
        sparsity_config["scheduler"]["kwargs"] = sparsity_config["scheduler"].get("kwargs", {})  # noqa E501
        sparse_scheduler = DEFAULT_REFERENCE_SCHEDULER_MAPPINGS[scheduler_type](**sparsity_config["scheduler"]["kwargs"])  # noqa E501
    else:
        custom_config_dict = {'MaskGeneratorConfig': sparsity_config["mask_generator"],  # noqa E501
                              'FakeSparseConfig': sparsity_config["fake_sparse"],  # noqa E501
                              'leaf_module': sparsity_config.get("leaf_module", []),
                              'fuse_bn': sparsity_config.get("fuse_bn", False),
                              'sparsity': sparsity_config.get("sparsity", None)}
        sparsity_config["scheduler"]["kwargs"] = sparsity_config["scheduler"].get("kwargs", {})  # noqa E501
        sparse_scheduler = DEFAULT_REFERENCE_SCHEDULER_MAPPINGS[scheduler_type](custom_config_dict, **sparsity_config["scheduler"]["kwargs"])  # noqa E501
    return sparse_scheduler
