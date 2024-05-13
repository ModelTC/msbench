
import torch
from msbench.scheduler.base_scheduler import Scheduler
from msbench.scheduler.nv_asp_cores import ASP


class AmpereScheduler(Scheduler):
    def __init__(self, mask_calculator="m4n2_1d", verbosity=2,
                 whitelist=[torch.nn.Linear, torch.nn.Conv2d],
                 allow_recompute_mask=False, allow_permutation=False):
        super().__init__()
        self.mask_calculator = mask_calculator
        self.verbosity = verbosity
        self.whitelist = whitelist
        self.allow_recompute_mask = allow_recompute_mask
        self.allow_permutation = allow_permutation

    def init_optimizer(self, optimizer):
        ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()

    def prepare_sparse_model(self, model):
        ASP.init_model_for_pruning(model, mask_calculator=self.mask_calculator,
                                   verbosity=self.verbosity,
                                   whitelist=self.whitelist,
                                   allow_recompute_mask=self.allow_recompute_mask,  # noqa E501
                                   allow_permutation=self.allow_permutation)
        return model

    def compute_sparse_rate(self, model):
        return 0.5
