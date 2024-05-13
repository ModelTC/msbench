import numpy as np
from msbench.utils.logger import logger
import torch

_SUPPORT_MODULE_TYPES = (torch.nn.Conv2d, torch.nn.Linear)


def update_sparsity_per_layer_from_sparsities(model, sparsities):
    zero_nums = 0
    total_nums = 0
    for name, m in model.named_modules():
        if isinstance(m, _SUPPORT_MODULE_TYPES):
            final_sparsity = sparsities[name]
            m.weight_fake_sparse.mask_generator.sparsity = final_sparsity
            zero_nums += final_sparsity * m.weight.numel()
            total_nums += m.weight.numel()
            logger.info('layer: {}, shape: {}, final sparsity: {}'.format(name, m.weight.shape, sparsities[name]))
    logger.info("After compute, real sparsity = {}".format(zero_nums / total_nums))


def get_st_sparsities_special(model, org_model):
    
    sparsities = {}
    state_dict = model.state_dict()
    org_state_dict = org_model.state_dict()

    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            org_weight = org_state_dict[name + ".weight"]
            cur_weight = state_dict[name + ".weight"]
            org_filter_num = org_weight.size(0)
            cur_filter_num = cur_weight.size(0)
            sparsities[name] = 1 - cur_filter_num / org_filter_num

    return sparsities


def get_st_sparsities_uniform(model, default_sparsity):
    
    sparsities = {}
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            if isinstance(module, torch.nn.Linear):
                sparsities[name] = 0.0
                sparsities[list(sparsities.keys())[-1]] = 0.0
            else:
                sparsities[name] = default_sparsity

    return sparsities


def get_unst_sparsities_uniform(model, default_sparsity):
    sparsities = {}
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            sparsities[name] = default_sparsity
    return sparsities


def get_sparsities_erdos_renyi(model,
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
        # logger.info('layer: {}, shape: {}, sparsity: {}'.format(name, module.weight.shape, sparsities[name]))
    return sparsities


def get_unst_sparsities_norm(model, default_sparsity, func='L2Normalized'):
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            if func == 'Magnitude':
                all_weights.append(torch.flatten(module.weight))
            elif func == 'L1Normalized':
                all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=1))
            elif func == 'L2Normalized':
                all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=2))
    all_weights = torch.cat(all_weights)
    all_weights = torch.absolute(all_weights)
    all_weights, _ = all_weights.sort()
    sparsity_threshold = all_weights[int(float(default_sparsity) * len(all_weights))]
    sparsities = {}
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            if func == 'Magnitude':
                mask = (torch.absolute(module.weight) > sparsity_threshold)
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
            elif func == 'L1Normalized':
                mask = (torch.absolute(module.weight / torch.norm(module.weight, p=1)) > sparsity_threshold)
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
            elif func == 'L2Normalized':
                mask = (torch.absolute(module.weight / torch.norm(module.weight, p=2)) > sparsity_threshold)
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
            sparsities[name] = sparsity
    return sparsities

def get_unst_sparsities_norm_v2(model, default_sparsity, max_sparsity, no_prune_layer_list, func='L2Normalized'):
    all_weights = []
    no_prun_layer_len = 0
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORT_MODULE_TYPES):
            if name in no_prune_layer_list:
                no_prun_layer_len += module.weight.numel()
                continue
            if func == 'Magnitude':
                all_weights.append(torch.flatten(module.weight))
            elif func == 'L1Normalized':
                all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=1))
            elif func == 'L2Normalized':
                all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=2))
    all_weights = torch.cat(all_weights)
    all_weights = torch.absolute(all_weights)
    all_weights, _ = all_weights.sort()
    sparsity_threshold = all_weights[int(float(default_sparsity) * (len(all_weights) + no_prun_layer_len))]
    sparsities = {}
    static_layer_name = []
    shift = 0
    search_flag = True
    while search_flag:
        search_flag = False
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                if name in no_prune_layer_list:
                    sparsities[name] = 0.0
                    continue
                if name in static_layer_name:
                    continue
                if func == 'Magnitude':
                    mask = (torch.absolute(module.weight) > sparsity_threshold)
                    sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
                elif func == 'L1Normalized':
                    mask = (torch.absolute(module.weight / torch.norm(module.weight, p=1)) > sparsity_threshold)
                    sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
                elif func == 'L2Normalized':
                    mask = (torch.absolute(module.weight / torch.norm(module.weight, p=2)) > sparsity_threshold)
                    sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
                if sparsity > max_sparsity:
                    static_layer_name.append(name)
                    search_flag = True
                    shift += (sparsity - max_sparsity) * module.weight.numel()
                    sparsity = max_sparsity
                sparsities[name] = sparsity
                # logger.info('layer: {}, shape: {}, sparsity: {}'.format(name, module.weight.shape, sparsities[name]))
        sparsity_threshold = all_weights[int(float(default_sparsity) * (len(all_weights) + no_prun_layer_len) + shift)]
    return sparsities

