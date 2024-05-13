from msbench.sparsity_mappings import SUPPORT_CONV, SUPPORT_LINEAR
from msbench.scheduler.base_scheduler import BaseScheduler


class STRScheduler(BaseScheduler):
    def __init__(self, custom_config_dict, no_prune_keyword='',
                 no_prune_layer=''):
        super().__init__(custom_config_dict, no_prune_keyword, no_prune_layer)

    def _apply_mask(self, model, mask_dict):
        for name, m in model.named_modules():
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                if name in self.no_prune_keyword + self.no_prune_layer:
                    continue
                else:
                    masked_weight = m.weight_fake_sparse.generate_sparse_weight(m.weight)  # noqa E501
                    m.weight.data.copy_(masked_weight)
