import argparse
from faulthandler import disable
from data.imagenet import load_data
from msbench.utils.state import disable_sparsification, enable_sparsification
from msbench.scheduler import build_sparse_scheduler
from msbench.advanced_pts import pts_reconstruction
from msbench.utils.logger import logger
from msbench.utils.dist_helper import setup_distributed, env, finalize
from msbench.utils.launch import launch
from models import load_model
from utils import parse_config, seed_all, evaluate
import logging
import time
import os
import torch
from msbench.advanced_pts import _SUPPORT_MODULE_TYPES
import numpy as np
from msbench.utils.global_flag import DIST_BACKEND

logger.setLevel(logging.INFO)

def read_sparse_analysis(file_path):
    sparse_matrix = dict()
    dict_prun_table_file = os.path.join(file_path, 'dict_prun_table.txt')
    desrie_prun_table_file = os.path.join(file_path, 'desrie_prun_table.txt')
    with open(dict_prun_table_file, 'r') as f:    # sparse ratio for each layer each stage
        sparse_matrix = eval(f.read())
    with open(desrie_prun_table_file, 'r') as f:  # desire pruning threshold and sparse ratio
        desrie_prun_table = f.read()
    desire_index_table = desrie_prun_table.splitlines()[4].replace("[", "").replace("]", "").split(",")
    desire_sparse_table = desrie_prun_table.splitlines()[5].replace("[", "").replace("]", "").split(",")
    desire_eng_table = desrie_prun_table.splitlines()[6].replace("[", "").replace("]", "").split(",")
    desire_index_table = list(map(int, desire_index_table))
    desire_sparse_table = list(map(float, desire_sparse_table))
    desire_eng_table = list(map(float, desire_eng_table))
    return sparse_matrix, desire_sparse_table, desire_index_table


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

def update_sparsity_per_layer(model, sparse_matrix, desire_energy_index):
    sparse_matrix_keys = list(sparse_matrix.keys())
    idx = 0
    zero_nums = 0
    total_nums = 0
    for name, m in model.named_modules():
        if isinstance(m, _SUPPORT_MODULE_TYPES):
            final_sparsity = sparse_matrix[sparse_matrix_keys[idx]][desire_energy_index]
            m.weight_fake_sparse.mask_generator.sparsity = final_sparsity
            idx += 1
            zero_nums += final_sparsity * m.weight.numel()
            total_nums += m.weight.numel()

def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    data_idx_range = range(env.rank * cali_batchsize, (env.rank + 1) * cali_batchsize) # If bs is too large, there is a bug.
    for i, batch in enumerate(train_loader):
        if i in data_idx_range:
            cali_data.append(batch[0])
        if i >= data_idx_range[-1]:
            break
    return cali_data


def progressive_pts_reconstruction(sparse_scheduler, model, cali_data, val_loader, config):
    prune_step = 0
    for sparsity_rate in config.sparse.progressive.sparsity_tables:
        get_sparsities(model, config, sparsity_rate)
        prune_step += 1
        logger.info("Current sparsity rate is {}".format(sparsity_rate))
        model = pts_reconstruction(model, cali_data, config.sparse.reconstruction)
        enable_sparsification(model)
        logger.info("Start to evaluate...")
        evaluate(val_loader, model)
        if hasattr(config, 'saver'):
            save_dir = config.saver.save_dir
            sparse_scheduler.export_sparse_model(model)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir = os.path.join(save_dir, config.model.type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sparse_ckpt_name = os.path.join(save_dir, '{}_global_sparse_rate_{}.pth.tar'.format(config.model.type, sparsity_rate))
            state = {}
            state['model'] = model.state_dict()
            state['sparsity_rate'] = sparsity_rate
            torch.save(state, sparse_ckpt_name)
            sparse_scheduler._enable_fake_sparse(model, True)

global sparse_matrix, desire_sparse_table, desire_index_table

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


def get_sparsities(model, config, sparsity):
    if hasattr(config.sparse, "sparse_analysis"):
        sparse_matrix, desire_sparse_table, desire_index_table = read_sparse_analysis(config.sparse.sparse_analysis)
        if sparsity:
            target_sparsity = sparsity
            prune_step = abs(np.array(desire_sparse_table) - target_sparsity).argmin()
            update_sparsity_per_layer(model, sparse_matrix, desire_index_table[prune_step])
            logger.info("set sparsity = {}".format(desire_sparse_table[prune_step]))
            logger.info("desire_sparse_table is {}".format(desire_sparse_table))
    elif hasattr(config.sparse, "set_sparsity"):
        if config.sparse.set_sparsity == "ERK":
            if hasattr(config.sparse, 'erk_power_scale'):
                erk_power_scale = config.sparse.erk_power_scale
            else:
                erk_power_scale = 1
            sparsities = get_sparsities_erdos_renyi(model, default_sparsity=sparsity, erk_power_scale=erk_power_scale)
            update_sparsity_per_layer_from_sparsities(model, sparsities)
        elif config.sparse.set_sparsity == "Magnitude":
            sparsities = get_unst_sparsities_norm(model, default_sparsity=sparsity, func='Magnitude')
            update_sparsity_per_layer_from_sparsities(model, sparsities)
        elif config.sparse.set_sparsity == "L2Normalized":
            sparsities = get_unst_sparsities_norm(model, default_sparsity=sparsity, func='L2Normalized')
            update_sparsity_per_layer_from_sparsities(model, sparsities)
        elif config.sparse.set_sparsity == "L2Normalized_v2":
            sparsities = get_unst_sparsities_norm_v2(model, default_sparsity=sparsity, max_sparsity=config.sparse.max_sparsity, no_prune_layer_list=config.sparse.no_prune_layer_list, func='L2Normalized')
            update_sparsity_per_layer_from_sparsities(model, sparsities)
        elif config.sparse.set_sparsity == "LAMP":
            pass



def main(args):
    config = parse_config(args.config)
    logger.info(config)
    # from torchvision.models import resnet
    # model = resnet.resnet18(pretrained=True)
    model = load_model(config.model)

    sparse_scheduler = build_sparse_scheduler(config.sparse)
    model = sparse_scheduler.prepare_sparse_model(model)

    model.cuda()
    
    # load_data
    train_loader, val_loader = load_data(**config.data)
    cali_data = load_calibrate_data(train_loader, cali_batchsize=config.sparse.cali_batchsize)

    logger.info('begin advanced PTS now!')
    start_time = time.time()
    model.train()
    
    if hasattr(config.sparse, "progressive"):
        progressive_pts_reconstruction(sparse_scheduler, model, cali_data, val_loader, config)
    else:
        sparsity = None
        if hasattr(config.sparse, "target_sparsity"):
            sparsity = config.sparse.target_sparsity
        get_sparsities(model, config, sparsity)
        if not hasattr(config.sparse, "mask_only"):
            model = pts_reconstruction(model, cali_data, config.sparse.reconstruction)
        else:
            logger.info('Mask only !!!')
        logger.info('after advanced PTS !')
        sparse_scheduler.export_sparse_model(model)
        disable_sparsification(model)
        evaluate(val_loader, model)
        sparsity_rate = sparse_scheduler.compute_sparse_rate(model)
        logger.info("sparsity rate is {}".format(sparsity_rate))
        if hasattr(config, 'saver'):
            save_dir = config.saver.save_dir
            exp_name = args.config.split('/')[-1][:-5]
            save_dir = os.path.join(save_dir, config.model.type, exp_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sparse_ckpt_name = os.path.join(save_dir, '{}_global_sparse_rate_{}.pth.tar'.format(config.model.type, sparsity_rate))
            state = {
                        'model': model.state_dict(),
                        'sparsity_rate': sparsity_rate
                    }
            torch.save(state, sparse_ckpt_name)
            sparse_scheduler.export_sparse_onnx(model, torch.randn(1, 3, 224, 224).cuda(), os.path.join(save_dir, '{}_global_sparse_rate_{}.onnx'.format(config.model.type, sparsity_rate)))
    finish_time = time.time()
    logger.info("pts_reconstruction use time {} mins".format((finish_time-start_time)/60))
    logger.info("pts done.")
    finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Solver')
    parser.add_argument('--config',
                        type=str,
                        default='configs/pst_res18_pr_50_layer_2w_1e-5_1k.yaml')
    parser.add_argument('--seed',
                        type=int,
                        default=1024)
    parser.add_argument('--launch',
                        dest='launch',
                        type=str,
                        default='slurm',
                        help='launch backend')
    parser.add_argument('--port',
                        dest='port',
                        type=int,
                        default=13333,
                        help='dist port')
    parser.add_argument('--backend',
                        dest='backend',
                        type=str,
                        default='dist',
                        choices=['linklink', 'dist'],
                        help='model backend')
    parser.add_argument('--ng', '--num_gpus_per_machine',
                        dest='num_gpus_per_machine',
                        type=int,
                        default=8,
                        help='num_gpus_per_machine')
    parser.add_argument('--nm', '--num_machines',
                        dest='num_machines',
                        type=int,
                        default=1,
                        help='num_machines')
    parser.add_argument('--fork-method',
                        dest='fork_method',
                        type=str,
                        default='fork',
                        choices=['spawn', 'fork'],
                        help='method to fork subprocess, especially for dataloader')
    args = parser.parse_args()
    seed_all(args.seed)
    DIST_BACKEND.backend = args.backend
    if args.launch == 'pytorch':
        launch(main, args.num_gpus_per_machine, args.num_machines, args=args, start_method=args.fork_method)
    else:
        setup_distributed(args.port, args.launch, args.backend)
        main(args)
