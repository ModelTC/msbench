from msbench.prepare import prepare
from msbench.sparsity_mappings import SUPPORT_CONV, SUPPORT_LINEAR
from msbench.utils.logger import logger
import numpy as np
import torch
from msbench.advanced_pts import _SUPPORT_MODULE_TYPES


class Scheduler:
    def __init__(self):
        pass

    def prepare_sparse_model(self, model):
        return model

    def prune_model(self, model, iteration):
        pass

    def _enable_fake_sparse(self, model, enabled: bool = True):
        pass

    def _generate_mask(self, model):
        pass

    def _apply_mask(self, model, mask_dict):
        pass

    def export_sparse_model(self, model):
        pass

    def export_sparse_onnx(self, model):
        pass

    def compute_sparse_rate(self, model):
        return None

    def update_sparsity_per_layer(self, model, sparsity_rate):
        pass


class BaseScheduler(Scheduler):
    def __init__(self, custom_config_dict, no_prune_keyword='',
                 no_prune_layer=''):
        super().__init__()
        self.custom_config_dict = custom_config_dict
        self.process_no_prune_layers(no_prune_keyword, no_prune_layer)

    def process_no_prune_layers(self, no_prune_keyword, no_prune_layer):
        self.no_prune_keyword = no_prune_keyword.split(",")
        self.no_prune_layer = no_prune_layer.split(",")
        for name in self.no_prune_keyword + self.no_prune_layer:
            if name.startswith('module.'):
                self.no_prune_keyword = [name.replace('module.', '', 1) if name.startswith('module.') \
                                         else name for name in self.no_prune_keyword]  # noqa E501
                self.no_prune_layer = [name.replace('module.', '', 1) if name.startswith('module.') \
                                        else name for name in self.no_prune_layer]  # noqa E501
                break

    def prepare_sparse_model(self, model):
        model = prepare(model, self.custom_config_dict)
        self._enable_fake_sparse(model, enabled=True)
        return model

    def prune_model(self, model, iteration):
        pass

    def _enable_fake_sparse(self, model, enabled: bool = True):
        for name, m in model.named_modules():
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                if name in self.no_prune_keyword + self.no_prune_layer:
                    m.weight_fake_sparse.enable_fake_sparse(False)
                else:
                    m.weight_fake_sparse.enable_fake_sparse(enabled)

    def _generate_mask(self, model):
        mask_dict = {}
        for name, m in model.named_modules():
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                mask = m.weight_fake_sparse.generate_mask(m.weight)
                mask_dict[name] = mask
        return mask_dict

    def _apply_mask(self, model, mask_dict):
        for name, m in model.named_modules():
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                if name in self.no_prune_keyword + self.no_prune_layer:
                    continue
                else:
                    mask = mask_dict[name].to(m.weight.device)
                    m.weight.data.mul_(mask)

    def export_sparse_model(self, model):
        mask_dict = self._generate_mask(model)
        self._apply_mask(model, mask_dict)
        self._enable_fake_sparse(model, False)

    def export_sparse_onnx(self, model, dummy_input, onnx_name):
        torch.onnx.export(model, dummy_input, onnx_name + ".onnx", verbose=False)

    def compute_sparse_rate(self, model):
        mask_dict = self._generate_mask(model)
        total_num = 0
        total_non_zero_num = 0
        for key, value in mask_dict.items():
            total_num += value.numel()
            total_non_zero_num += value.count_nonzero()
        sparse_rate = 1 - (total_non_zero_num / total_num).item()
        return sparse_rate

    def update_sparsity_per_layer(self, model, sparsity_rate):
        for idx, (name, m) in enumerate(model.named_modules()):
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                m.weight_fake_sparse.mask_generator.sparsity = sparsity_rate

    def get_sparsities_erdos_renyi(self, model,
                                   default_sparsity,
                                   custom_sparsity_map=[],
                                   include_kernel=True,
                                   erk_power_scale=0.5):
        def get_n_zeros(size, sparsity):
            return int(np.floor(sparsity * size))

        fp32_modules = dict()
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                fp32_modules[name] = module

        is_eps_valid = False
        dense_layers = set()

        while not is_eps_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, module in fp32_modules.items():
                shape_list = list(module.weight.shape)
                n_param = np.prod(shape_list)
                n_zeros = get_n_zeros(n_param, default_sparsity)
                if name in dense_layers:
                    rhs -= n_zeros
                elif name in custom_sparsity_map:
                    # We ignore custom_sparsities in erdos-renyi calculations.
                    pass
                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    n_ones = n_param - n_zeros
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    if include_kernel:
                        raw_probabilities[name] = (np.sum(shape_list) / np.prod(shape_list))**erk_power_scale
                    else:
                        n_in, n_out = shape_list[-2:]
                        raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            eps = rhs / divisor
            # If eps * raw_probabilities[name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * eps
            if max_prob_one > 1:
                is_eps_valid = False
                for name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        logger.info('Sparsity of var: {} had to be set to 0.'.format(name))
                        dense_layers.add(name)
            else:
                is_eps_valid = True
            # exit()
        sparsities = {}
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, module in fp32_modules.items():
            shape_list = list(module.weight.shape)
            n_param = np.prod(shape_list)
            if name in custom_sparsity_map:
                sparsities[name] = custom_sparsity_map[name]
                logger.info('layer: {} has custom sparsity: {}'.format(name, sparsities[name]))
            elif name in dense_layers:
                sparsities[name] = 0
            else:
                probability_one = eps * raw_probabilities[name]
                sparsities[name] = 1. - probability_one
            logger.info('layer: {}, shape: {}, sparsity: {}'.format(name, module.weight.shape, sparsities[name]))

        zero_nums = 0
        total_nums = 0
        for name, m in model.named_modules():
            if isinstance(m, _SUPPORT_MODULE_TYPES):
                final_sparsity = sparsities[name]
                m.weight_fake_sparse.mask_generator.sparsity = final_sparsity
                zero_nums += final_sparsity * m.weight.numel()
                total_nums += m.weight.numel()
        logger.info("After compute, real sparsity = {}".format(zero_nums / total_nums))
        return sparsities
