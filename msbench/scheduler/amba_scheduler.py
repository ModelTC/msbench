from msbench.scheduler.base_scheduler import BaseScheduler
from msbench.sparsity_mappings import SUPPORT_CONV, SUPPORT_LINEAR
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import subprocess
from collections import OrderedDict

path_share_lib = os.path.dirname(os.path.realpath(__file__)) + '/amba_lib/'
# Here, add the amba_lib path.
if os.path.exists(path_share_lib):
    sys.path.append(path_share_lib)
try:
    from version_check import check_criterion_version
    from hw_analysis import compute_target_density, get_min_split_kernel_size, get_dilated_kernel_size
    check_criterion_version(0x20)
except:  # noqa: E722
    print('[ERROR] Please make sure PYTHONPATH includes amba_lib/')
    exit(1)


def convert_pair(x):
    if len(x) == 1:
        x = (x[0], x[0])
    return x


def is_dist_avail_and_initialized():
    if USE_LINKLINK:
        if not link.is_initialized():
            return False
        return True
    else:
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if USE_LINKLINK:
        return link.get_rank()
    elif not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class AmbaLevelPruneScheduler(BaseScheduler):
    def __init__(self, custom_config_dict, total_iters=10000, sparsity_table=[30, 40, 50, 60, 70, 80, 90], no_prune_keyword='',
                 no_prune_layer='', prun_algo=1, prun_algo_tuning=0.5, prun_hw_aware='', dw_no_prune=False,
                 do_sparse_analysis=True, output_dir='./sparse_analysis', save_dir='./model_save_path', sparse_rate_tolerance=0.015,
                 default_input_shape=None, default_input_custom=None, lr_scheduler=None, mask_dict=None,
                 prune_tolerance_max=1.0, prune_tolerance_min=1.0, init_lr=None, verbose=False):
        super().__init__(custom_config_dict, no_prune_keyword, no_prune_layer)
        self.custom_config_dict = custom_config_dict
        self.process_no_prune_layers(no_prune_keyword, no_prune_layer)

        self.bin_core_sparse_ratio_decision = os.path.join(os.path.realpath(path_share_lib), 'core_sparse_ratio_decision')
        assert len(sparsity_table) > 0, "sparsity cannot be None"
        self.sparse_table = [s / 100. for s in sparsity_table]
        self.total_iters = total_iters
        # self.iters = [int(n * iters_per_stage) for n in range(len(sparsity_table))]
        self.iters = [int(n * total_iters // len(sparsity_table)) for n in range(len(sparsity_table))]
        if get_rank() == 0:
            print('sparsity steps: ' + str(self.sparse_table))
            print('iterations that will change the sparsity: ' + str(self.iters))

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler  # if lr_scheduler is given, it will be reset for every prune step
        else:
            self.lr_scheduler = None

        self.mask_dict = mask_dict  # you can resume the mask dict here, otherwise it's initialized to None
        # Resume
        if self.mask_dict is not None:
            self.resume_en = True
        else:
            self.resume_en = False

        self.prun_algo = prun_algo
        # prun_algo: 0 is just to consider energy_th as pruning ratio directly so all layers will share the same pruning ratio. #uniform#  # noqa: E501
        # prun_algo: 1 is advanced version. PRUN_ALGO_TUNING is to control decay slope
        self.prun_algo_tuning = prun_algo_tuning
        self.prun_hw_aware = prun_hw_aware
        self.prun_hw_aware_en = len(prun_hw_aware) > 0
        self.prune_tolerance_max = prune_tolerance_max
        self.prune_tolerance_min = prune_tolerance_min
        self.dw_no_prune = dw_no_prune
        self.sparse_rate_tolerance = sparse_rate_tolerance
        self.mask_from_prune_step = -1
        self.init_iter = None
        self.init_lr = init_lr
        self.nn_analysis_speedup = 0
        self.iteration = 0
        self.verbose = verbose
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Pruning analysis
        self.do_sparse_analysis = do_sparse_analysis
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        self.dict_prun_table_file = os.path.join(output_dir, 'dict_prun_table.txt')  # store sparse ratio for each layer
        self.desrie_prun_table_file = os.path.join(output_dir, 'desrie_prun_table.txt')  # store sparse ratio for each layer
        self.hw_aware_sparsity_table_file = os.path.join(output_dir, 'hw_aware_sparsity_table_{}.txt'.format(self.prun_hw_aware))

        # Summary file
        self.pruner_report_file = os.path.join(output_dir, 'pruner_report.csv')
        self.pruner_statistics = None

        # input of user's NN
        self.default_input_shape = default_input_shape
        self.default_input_custom = default_input_custom

    def _sparsity_function(self, s_i, s_f, total_iter, n_steps, t_start=0, t_end=-1):
        if t_end == -1:
            t_end = total_iter
        if t_end > 0 and t_end > total_iter:
            if get_rank() == 0:
                print('prune_ending_iter is larger than total_iter, please check if it is intended')
        s_range = s_f - s_i
        delta_iter = int((t_end - t_start) / float(n_steps))
        iters = [delta_iter * n + t_start for n in range(n_steps)]
        sparse_table = [s_f - (s_range * np.power(1 - ((t - t_start) / float((n_steps - 1) * delta_iter)), 3)) for t in iters]
        return sparse_table, iters

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def prune_model(self, model, iteration, curr_input_shape=None):  # noqa: F401
        '''
        param: model: pytorch model
        param: curr_input_shape: input shape of the model. This will be used to calculate the MAC.
                                This can be different from default_input_shape.
                                (e.g., for image input, you can set (1, 3, H, W))
        param: iteration: should be zero indexing (first iteration is 0)
        '''
        self.iteration = iteration
        prune_step = 0
        # input_shape
        if (self.default_input_shape is None) and (self.default_input_custom is None) and (curr_input_shape is None):
            if get_rank() == 0:
                print('Please specify one of arguments: default_input_shape, default_input_custom and curr_input_shape')
                print('default_input_shape and default_input_custom can be specified in Pruner class')
                print('curr_input_shape can be specified in prune_model function')
            sys.exit(1)
        input_shape = self.default_input_shape if curr_input_shape is None else curr_input_shape

        if self.prun_algo == 1:
            if os.path.exists(self.bin_core_sparse_ratio_decision) is False:
                if get_rank() == 0:
                    print('The file is not exist. Please check path {}'.format(self.bin_core_sparse_ratio_decision))
                sys.exit(1)

        # first iteration
        if self.init_iter is None:
            self.init_iter = iteration
            if self.do_sparse_analysis:
                self.sparse_analysis(model, input_shape)
            else:
                if not os.path.exists(self.dict_prun_table_file) or not os.path.exists(self.desrie_prun_table_file) or \
                   (self.prun_hw_aware_en and not os.path.exists(self.hw_aware_sparsity_table_file)):
                    self.sparse_analysis(model, input_shape)
            self.sparse_matrix, _, self.desire_index_table, self.hw_aware_sparsity_table = self._read_sparse_analysis()
            if get_rank() == 0:
                print("sparse_matrix: ", self.sparse_matrix)
                print("desire_index_table: ", self.desire_index_table)
                if self.hw_aware_sparsity_table:
                    print("hw_aware_sparsity_table: ", self.hw_aware_sparsity_table)

            # get current statistics first
            prune_step = len([i for i in self.iters if i < iteration]) - 1
            if prune_step >= 0 and self.mask_dict is None:  # generate mask for current prune step
                self.mask_from_prune_step = prune_step
                self._update_sparsity_per_layer(model, self.sparse_matrix, self.desire_index_table[prune_step],
                                                self.hw_aware_sparsity_table)

        if iteration in self.iters:
            # generate new mask
            prune_step = self.iters.index(iteration)
            self.mask_from_prune_step = prune_step
            self._update_sparsity_per_layer(model, self.sparse_matrix, self.desire_index_table[prune_step],
                                            self.hw_aware_sparsity_table)
            # reset the lr scheduler
            if self.lr_scheduler is not None and (iteration > 0):
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                    self.lr_scheduler.step(epoch=0)
                elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler._reset()
                    if not self.init_lr:
                        if get_rank() == 0:
                            print('ReduceLROnPlateau needs init_lr(initial learning rate,\
                               float or list of float) to reset learning rate')
                        sys.exit(1)
                    if len(self.lr_scheduler.optimizer.param_groups) == 1:
                        self.lr_scheduler.optimizer.param_groups[0]['lr'] = self.init_lr
                    else:
                        for i, param_group in enumerate(self.lr_scheduler.optimizer.param_groups):
                            self.lr_scheduler.optimizer.param_groups[i]['lr'] = self.init_lr[i]
                else:  # for prototype defined LRScheduler
                    self.lr_scheduler.step(this_iter=0)
        else:
            if self.mask_from_prune_step == -1:
                prune_step = len([i for i in self.iters if i < iteration]) - 1
                self.mask_from_prune_step = prune_step
                self._update_sparsity_per_layer(model, self.sparse_matrix, self.desire_index_table[prune_step],
                                                self.hw_aware_sparsity_table)
        decision_iters = self.iters + [self.total_iters]
        if iteration + 1 in decision_iters:
            prune_step = decision_iters.index(iteration + 1) - 1
            if get_rank() == 0:
                print("iteration = ", iteration)
                print("save_ckpts....")
                print("sparse_table: ", self.sparse_table)
                print("prune_step: ", prune_step)
            self.export_sparse_model(model)
            sparse_ckpt_name = os.path.join(self.save_dir, 'global_sparse_rate_{}.pth.tar'.format(self.sparse_table[prune_step]))
            self.state = {}
            self.state['model'] = model.state_dict()
            self.state['prune_step'] = prune_step
            self.state['sparse_matrix'] = self.sparse_matrix
            self.state['desire_index_table'] = self.desire_index_table
            if get_rank() == 0:
                torch.save(self.state, sparse_ckpt_name)
            self._enable_fake_sparse(model, True)

    def get_prune_step(self):
        if self.iteration == 0:
            return self.iteration
        else:
            prune_step = len([i for i in self.iters if i < self.iteration]) - 1
            # return prune_step
            return self.iters[prune_step]

    def get_last_prune_step(self):
        if self.iteration == 0:
            return self.iteration
        else:
            prune_step = len([i for i in self.iters if i < self.iteration]) - 1
            # return prune_step
            return self.iters[prune_step]

    def _update_sparsity_per_layer(self, model, sparse_matrix, desire_energy_index, hw_aware_sparsity_table):
        for idx, (name, m) in enumerate(model.named_modules()):
            if name not in sparse_matrix:
                continue
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                final_sparsity = sparse_matrix[name][desire_energy_index]
                if self.prun_hw_aware_en:
                    final_sparsity = min(final_sparsity, hw_aware_sparsity_table[name][0])
                m.weight_fake_sparse.mask_generator.sparsity = final_sparsity

    def show_statistics(self, model, input_shape):  # noqa: F401
        def register_hook(module):
            def hook(module, input, output):

                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                batch_size = -1
                m_key = "%s-%i" % (class_name, module_idx + 1)

                summary[m_key] = OrderedDict()
                summary[m_key]['name'] = m_key
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                nonzeros = 0
                mac = 0
                mac_remain = 0

                # params
                # if hasattr(module, "weight") and hasattr(module.weight, "size"):
                if isinstance(module, SUPPORT_CONV) or isinstance(module, SUPPORT_LINEAR):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    nonzeros += np.count_nonzero(module.weight.cpu().data)
                    summary[m_key]["trainable"] = module.weight.requires_grad
                    summary[m_key]["weight"] = module.weight
                summary[m_key]["nb_params"] = params
                summary[m_key]["nonzero_params"] = nonzeros

                # mac
                if isinstance(module, SUPPORT_CONV):
                    # input tuple(tensor(B, C, H, W))
                    # output tensor(B, C, H, W)
                    mac = params * output.size()[2] * output.size()[3]
                    mac_remain = mac * nonzeros // params
                    layer_info[m_key] = OrderedDict()
                    layer_info[m_key]['name'] = m_key
                    layer_info[m_key]["input_shape"] = list(input[0].size())
                    layer_info[m_key]["input_shape"][0] = batch_size
                    if isinstance(output, (list, tuple)):
                        layer_info[m_key]["output_shape"] = [
                            [-1] + list(o.size())[1:] for o in output
                        ]
                    else:
                        layer_info[m_key]["output_shape"] = list(output.size())
                        layer_info[m_key]["output_shape"][0] = batch_size
                elif isinstance(module, SUPPORT_LINEAR):
                    # input tuple(tensor(B, W))
                    # output tensor(B, W)
                    mac = input[0].size()[1] * output.size()[1]
                    mac_remain = mac * nonzeros // params
                    layer_info[m_key] = OrderedDict()
                    layer_info[m_key]['name'] = m_key
                    layer_info[m_key]["input_shape"] = list(input[0].size())
                    layer_info[m_key]["input_shape"][0] = batch_size
                    if isinstance(output, (list, tuple)):
                        layer_info[m_key]["output_shape"] = [
                            [-1] + list(o.size())[1:] for o in output
                        ]
                    else:
                        layer_info[m_key]["output_shape"] = list(output.size())
                        layer_info[m_key]["output_shape"][0] = batch_size
                summary[m_key]["MAC"] = mac
                summary[m_key]["MAC_remain"] = mac_remain
            if(isinstance(module, SUPPORT_CONV) or  # noqa: W504
               isinstance(module, SUPPORT_LINEAR) or  # noqa: W504
               isinstance(module, nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(hook))

        def fix_batch_static(m):
            if type(m) == nn.BatchNorm2d:
                m.track_running_stats = False

        def update_batch_static(m):
            if type(m) == nn.BatchNorm2d:
                m.track_running_stats = True

        def module_training_status(model, isRead):
            if isRead:
                self.module_training_status = dict()
                for idx, (name, m) in enumerate(model.named_modules()):
                    self.module_training_status[name] = m.training
            else:
                if len(self.module_training_status) == 0:
                    if get_rank() == 0:
                        self.log.error('Please make sure to run module_training_status with isRead=True before')
                for idx, (name, m) in enumerate(model.named_modules()):
                    m.training = self.module_training_status[name]
        summary = OrderedDict()
        layer_info = OrderedDict()
        hooks = []

        # remove the DP and DDP wrap
        # model = remove_module(model)

        model.apply(register_hook)
        # model.apply(fix_batch_static)

        module_training_status(model, isRead=True)
        model.eval()
        if self.default_input_custom is None:
            dtype = torch.cuda.FloatTensor if next(model.parameters()).is_cuda else torch.FloatTensor
            input_shape_DHW = tuple(input_shape[1:4])
            # x = torch.rand(2, *input_shape_DHW).type(dtype)
            x = {"image": torch.rand(2, *input_shape_DHW).type(dtype), "label": torch.randn(1)}
            # input shape should be (2, D, H, W) if there has batch normlization layer
            model(x)
        else:
            model(self.default_input_custom)
        # model.apply(update_batch_static)
        module_training_status(model, isRead=False)

        for h in hooks:
            h.remove()

        module_name_map = {}
        for idx, (name, m) in enumerate(model.named_modules()):
            if isinstance(m, SUPPORT_CONV) or isinstance(m, SUPPORT_LINEAR):
                for k, v in summary.items():
                    if 'weight' in v:
                        if torch.equal(v['weight'], m.weight):
                            module_name_map[name] = k

        total_param = 0
        total_nonzero = 0
        total_mac = 0
        total_mac_remain = 0
        for layer in summary:
            if get_rank() == 0 and self.verbose:
                print('layer: {}, output: {}, input: {}'.format(layer, summary[layer]["output_shape"],
                                                                summary[layer]["input_shape"]))
            total_param += summary[layer]["nb_params"]
            total_nonzero += summary[layer]["nonzero_params"]
            total_zero = total_param - total_nonzero
            total_mac += summary[layer]["MAC"]
            total_mac_remain += summary[layer]["MAC_remain"]
        total_param = total_param.item()
        total_zero = total_zero.item()
        total_mac = total_mac.item()
        total_mac_remain = total_mac_remain.item()
        if get_rank() == 0:
            print('{:17} {:17} {:17} {:17} {:17} {:17}'.format('total_param', 'total_nonzero',
                                                               'sparsification', 'total_mac',
                                                               'total_mac_remain', 'mac_remain_ratio'))
            print('{:<17d} {:<17d} {:<17f} {:<17d} {:<17d} {:<17f}'.format(total_param,
                                                                           total_nonzero,
                                                                           1.0 * total_zero / total_param,
                                                                           total_mac,
                                                                           total_mac_remain,
                                                                           1.0 * total_mac_remain / total_mac))
        return (total_param, total_nonzero, 1.0 * total_zero / total_param, total_mac, total_mac_remain,
                1.0 * total_mac_remain / total_mac), layer_info, module_name_map

    def _read_sparse_analysis(self):
        if is_dist_avail_and_initialized():
            if USE_LINKLINK:
                link.barrier()
            else:
                dist.barrier()

        sparse_matrix = dict()
        hw_aware_sparsity_table = dict()
        with open(self.dict_prun_table_file, 'r') as f:    # sparse ratio for each layer each stage
            sparse_matrix = eval(f.read())
        with open(self.desrie_prun_table_file, 'r') as f:  # desire pruning threshold and sparse ratio
            desrie_prun_table = f.read()
        desire_index_table = desrie_prun_table.splitlines()[4].replace("[", "").replace("]", "").split(",")
        desire_sparse_table = desrie_prun_table.splitlines()[5].replace("[", "").replace("]", "").split(",")
        desire_eng_table = desrie_prun_table.splitlines()[6].replace("[", "").replace("]", "").split(",")
        desire_index_table = list(map(int, desire_index_table))
        desire_sparse_table = list(map(float, desire_sparse_table))
        desire_eng_table = list(map(float, desire_eng_table))
        if self.prun_hw_aware_en:
            with open(self.hw_aware_sparsity_table_file, 'r') as f:    # sparse ratio for each layer each stage
                hw_aware_sparsity_table = eval(f.read())
        return sparse_matrix, desire_sparse_table, desire_index_table, hw_aware_sparsity_table

    def sparse_analysis(self, model, input_shape):  # noqa: F401
        if not is_main_process():
            return 0
        test_index_table = []  # store index for all results from derivation
        test_sparse_table = []  # store sparsification for all results from derivation
        desire_index_table = []  # store index for desire results from derivation
        desire_sparse_table = []  # store sparsification for desire results from derivation
        desire_eng_table = []  # store corresponding energy for desire sparsification
        weight_info_dict = {}
        max_energy_tmp = 10
        min_energy_tmp = 0
        max_energy = None
        min_energy = None
        if self.resume_en is False:
            if get_rank() == 0:
                print('Network analysis process START')
            tmp_matrix = dict()
            init_energy = max_energy_tmp  # inital energy
            # energy_increment = 0
            curr_energy_index = 0
            curr_energy = init_energy
            prev_energy = 0
            curr_sparse = 0
            prev_sparse = 0
            # check sparsification before perform doing process
            init_sparse, layer_info, module_name_map = self.show_statistics(model, input_shape)

            if self.nn_analysis_speedup:
                self.sparse_table = [self.sparse_table[0], self.sparse_table[-1]]

            while len(desire_index_table) < len(self.sparse_table):
                b_achieve_net_max_sparse = []
                if self.prun_algo == 0:
                    curr_energy = self.sparse_table[len(desire_index_table)]
                curr_sparse = self._parameter_hist(weight_info_dict, model, curr_energy,
                                                   self.sparse_table[len(desire_index_table)],
                                                   tmp_matrix, b_achieve_net_max_sparse,
                                                   self.prun_algo, self.prun_algo_tuning,
                                                   self.no_prune_keyword, self.no_prune_layer,
                                                   self.bin_core_sparse_ratio_decision,
                                                   layer_info, module_name_map)
                if self.prun_algo == 0:
                    desire_index_table.append(curr_energy_index)
                    desire_sparse_table.append(curr_sparse)
                    desire_eng_table.append(curr_energy)
                else:
                    test_index_table.append(curr_energy_index)
                    test_sparse_table.append(curr_sparse)
                    diff_energy = curr_energy - prev_energy
                    diff_sparse = curr_sparse - prev_sparse
                    if get_rank() == 0:
                        print('{:12} {:12} {:12} {:12} {:14} {:12} {:17} {:19}'.format('curr_energy', 'curr_sparse',
                                                                                       'prev_energy', 'prev_sparse',
                                                                                       'target_sparse', 'diff_energy',
                                                                                       'diff_prev_sparse',
                                                                                       'diff_target_sparse'))
                        print('{:<12f} {:<12f} {:<12f} {:<12f} {:<14f} {:<12f} {:<17f} {:<19f}'.format(curr_energy,
                                                                                                       curr_sparse,
                                                                                                       prev_energy, prev_sparse,  # noqa: E501
                                                                                                       self.sparse_table[len(desire_index_table)],  # noqa: E501
                                                                                                       diff_energy, diff_sparse,  # noqa: E501
                                                                                                       self.sparse_table[len(desire_index_table)] - curr_sparse))  # noqa: E501

                    if abs(curr_sparse - self.sparse_table[len(desire_index_table)]) <= self.sparse_rate_tolerance or\
                       (max_energy and min_energy and ((max_energy - min_energy) < 1e-6)):
                        # find the appropriate energy, find next
                        desire_index_table.append(curr_energy_index)
                        desire_sparse_table.append(curr_sparse)
                        desire_eng_table.append(curr_energy)
                        min_energy, max_energy = None, None  # start from a new search

                    if len(desire_index_table) < len(self.sparse_table):
                        if curr_sparse > self.sparse_table[len(desire_index_table)]:
                            max_energy = curr_energy
                            max_energy_tmp = curr_energy
                            if max_energy - min_energy_tmp <= 1:
                                min_energy_tmp = max_energy - 10 if min_energy is None else max(min_energy, max_energy - 10)
                        else:
                            min_energy = curr_energy
                            min_energy_tmp = curr_energy
                            if max_energy_tmp - min_energy <= 1:
                                max_energy_tmp = min_energy + 10 if max_energy is None else min(max_energy, min_energy + 10)

                    prev_energy = curr_energy
                    prev_sparse = curr_sparse
                    curr_energy = (max_energy_tmp + min_energy_tmp) / 2.0  # max(0, curr_energy + energy_increment)
                curr_energy_index = curr_energy_index + 1
                if b_achieve_net_max_sparse[0]:
                    if get_rank() == 0 and self.verbose:
                        print('Analysis is terminated since network already achieves maximum sparsification {}\
                                       of user''s configuration'.format(curr_sparse))
                    # break
            if self.nn_analysis_speedup:
                # new_tmp_matrix = []
                init_spars_idx = desire_index_table[0]
                end_spars_idx = desire_index_table[1]
                desire_index_table = list(range(0, self.prune_steps))
                self.sparse_table, _ = self._sparsity_function(self.sparse_table[0], self.sparse_table[-1], self.total_iter,   # noqa E501
                                                               self.prune_steps, self.prune_start_iter, self.prune_ending_iter)  # noqa E501
                for name in tmp_matrix.keys():
                    init_spars = tmp_matrix[name][init_spars_idx]
                    end_spars = tmp_matrix[name][end_spars_idx]
                    tmp_matrix[name], _ = self._sparsity_function(init_spars, end_spars,  # noqa E501
                                                                  self.total_iter,  # noqa E501
                                                                  self.prune_steps,  # noqa E501
                                                                  self.prune_start_iter,  # noqa E501
                                                                  self.prune_ending_iter)  # noqa E501

            with open(self.dict_prun_table_file, 'w') as f:
                f.write(str(tmp_matrix))
            with open(self.desrie_prun_table_file, 'w') as f:
                f.write('search table, index and network_sparsification\n')
                f.write(str(test_index_table) + '\n')
                f.write(str(test_sparse_table) + '\n')
                f.write('desire table, index, network sparsification and network energy\n')  # noqa E501
                f.write(str(desire_index_table) + '\n')
                f.write(str(desire_sparse_table) + '\n')
                f.write(str(desire_eng_table) + '\n')
            if self.prun_hw_aware_en:
                with open(self.hw_aware_sparsity_table_file, 'w') as f:
                    hw_aware_sparsity_dict = dict()
                    for name in weight_info_dict.keys():
                        self._build_dict(name, weight_info_dict[name]['hw_aware_sparsity'], hw_aware_sparsity_dict)  # noqa E501
                    f.write(str(hw_aware_sparsity_dict))
            if get_rank() == 0:
                print('Network analysis process DONE')

    def _build_dict(self, name, value, self_matrix):  # build dict codebook
        if (not(name in self_matrix)):
            self_matrix.update({name: [value]})
        else:
            self_matrix[name].append(value)

    def _parameter_hist(self, weight_info_dict, model, curr_energy,
                        target_nn_sparse, self_sparse_matrix,
                        b_achieve_net_max_sparse, prun_algo,
                        prun_algo_tuning, no_prun_keyword,
                        no_prun_layer,
                        bin_core_sparse_ratio_decision,
                        layer_info, module_name_map):
        # network analysis: generate desired pruning parameters
        # based on curr_energy to derive final energy for layers within a network  # noqa E501
        # return estimated network sparsification
        MAX_SPARSE = 0.99
        # get stride group for each layer
        conv_list, conv_indx, conv_strd = self._stride_group_derivation(model, layer_info, module_name_map)  # noqa E501
        # print_log conv_strd

        total_param = 0  # STAT: total coeffcieitns for a network
        total_nonzero = 0  # STAT: non-zero coefficients for a network
        total_zero = 0  # STAT: zero coefficients for a network

        net_sparse_curr = 0
        net_sparse_max = 0

        for idx, (name, m) in enumerate(model.named_modules()):
            if name not in module_name_map:
                continue
            tmp_energy_th = curr_energy  # current energy
            layer_param = 0  # STAT: total coefficients for a layer
            sparse_ratio = 0  # STAT: zero coefficients for a layer
            max_stride_group = max(conv_strd)
            stride_group = 0
            ctrl_bits = 0
            bFlag = 0

            # no pruning for specific layers which satisfy user-dfined NO_PRUN_KEYWORD and NO_PRUN_LAYER  # noqa E501
            if(len(no_prun_keyword) > 0):
                for i in range(len(no_prun_keyword)):
                    if len(no_prun_keyword[i]) > 0 and (name.find(no_prun_keyword[i]) >= 0):  # noqa E501
                        tmp_energy_th = 0
            if(len(no_prun_layer) > 0):
                # self.log.info(name)
                if(name in no_prun_layer):
                    tmp_energy_th = 0

            # so far we support convolution and fully connected
            if (isinstance(m, SUPPORT_CONV)):
                out_c, in_c, w, h = m.weight.shape
                layer_param = out_c * in_c * w * h
                stride_group = conv_strd[conv_list.index(name)]
                is_dw = 1 if (m.groups != 1) else 0
                if get_rank() == 0 and self.verbose:
                    print('{}: stride_group: {} is_dw: {}'.format(name, stride_group, is_dw))  # noqa E501
                bFlag = 1
            elif (isinstance(m, SUPPORT_LINEAR)):
                # pruning for fully conntected has an assumption.
                # Ths assumption is FC layer at last layer so stride_group is always equal to max_stride_group  # noqa E501
                out_c, in_c = m.weight.shape
                layer_param = out_c * in_c
                stride_group = max_stride_group
                if get_rank() == 0 and self.verbose:
                    print('{}: stride_group: {} is_dw: {}'.format(name, stride_group, is_dw))  # noqa E501
                is_dw = 0
                bFlag = 1

            if (bFlag == 1):
                self._get_weight_information(weight_info_dict, name, m)
                sparse_ratio = self._sparse_ratio_decision(tmp_energy_th,
                                                           stride_group,
                                                           max_stride_group,
                                                           name, ctrl_bits,
                                                           is_dw, prun_algo,
                                                           prun_algo_tuning,
                                                           bin_core_sparse_ratio_decision,  # noqa E501
                                                           weight_info_dict)
                if self.prune_tolerance_max > 1.0:
                    layer_max_sparsity = target_nn_sparse * self.prune_tolerance_max  # noqa E501
                    sparse_ratio = min(sparse_ratio, layer_max_sparsity)
                if self.prune_tolerance_min < 1.0:
                    target_nn_density = 1 - target_nn_sparse
                    layer_min_density = target_nn_density * self.prune_tolerance_min  # noqa E501
                    layer_max_sparsity = 1.0 - layer_min_density
                    sparse_ratio = min(sparse_ratio, layer_max_sparsity)
                if is_dw and self.dw_no_prune:
                    sparse_ratio = 0.0
                if get_rank() == 0 and self.verbose:
                    print('pruning ratio for {} is {}'.format(name, sparse_ratio))  # noqa E501
                self._build_dict(name, sparse_ratio, self_sparse_matrix)
                net_sparse_curr += sparse_ratio
                net_sparse_max += MAX_SPARSE if tmp_energy_th else 0

            # STAT
            total_param += layer_param
            total_zero += layer_param * sparse_ratio
            # print_log sparse_ratio, l.name, layer_param, layer_param*sparse_ratio  # noqa E501
        # STAT
        total_nonzero = total_param - total_zero  # noqa: F841
        network_sparsification = 1.0 * total_zero / total_param
        if (net_sparse_curr >= net_sparse_max) and (curr_energy > 0):
            b_achieve_net_max_sparse.append(True)
        else:
            b_achieve_net_max_sparse.append(False)
        return network_sparsification

    def _stride_group_derivation(self, model, layer_info, module_name_map):
        # stride group is for layer group derivation
        # return layer group information
        layer_list = []  # noqa: F841
        # store layers which are included in net_par
        # layer_info = list(layer_info.items())

        conv_list = []  # store layers which have weights
        conv_indx = []  # store corresponding index in net_par for conv_list
        conv_strd = []  # sotre times of down-sampling for conv_list
        init_input_size = max(list(layer_info.values())[0]['input_shape'][2], list(layer_info.values())[0]['input_shape'][3])  # noqa E501
        input_list = []
        import math
        for idx, (name, m) in enumerate(model.named_modules()):
            if name not in module_name_map:
                continue
            # store layers which have weights
            if isinstance(m, SUPPORT_CONV):
                layer_name = module_name_map[name]
                curr_input_size = max(layer_info[layer_name]['input_shape'][2], layer_info[layer_name]['input_shape'][3])  # noqa E501
                conv_list.append(name)
                conv_indx.append(idx)
                conv_strd.append(math.ceil(math.log(1.0 * init_input_size / curr_input_size, 2)))  # noqa E501
                input_list.append(curr_input_size)
            elif isinstance(m, SUPPORT_LINEAR):
                layer_name = module_name_map[name]
                conv_list.append(name)
                conv_indx.append(idx)
                conv_strd.append(0)
                input_list.append(0)
        if get_rank() == 0 and self.verbose:
            print('len(conv_list) {} len(layer_info) {}'.format(len(conv_list), len(layer_info)))  # noqa E501

        # [5,5,3,0,7,10] -> [2,2,1,0,3,4]
        group_unique = sorted(list(set(conv_strd)))
        group_index = list(range(len(group_unique)))
        conv_strd = [group_index[group_unique.index(x)] for x in conv_strd]

        if get_rank() == 0 and self.verbose:
            print(str(conv_strd))
            print(str(input_list))

        for i in range(len(conv_list)):
            if get_rank() == 0 and self.verbose:
                print('{} {} {}'.format(conv_list[i], conv_strd[i], input_list[i]))  # noqa E501
        return conv_list, conv_indx, conv_strd

    def _get_weight_information(self, weight_info_dict, name, m):
        if name in weight_info_dict:
            return
        else:
            w = m.weight.data
            w_flat = w.flatten().cpu()
            len_w = len(w_flat)
            w_sorted_abs = np.sort(abs(w_flat))
            zero_percent = (len_w - len(np.nonzero(w_flat))) / len_w * 100.0

            w_square_sorted = np.power(w_sorted_abs, 2)
            energy_sum = np.sum(w_square_sorted)
            energy_cummulated = np.cumsum(w_square_sorted / energy_sum)
            hw_aware_sparsity = self._get_target_sparsity(name, m)

            weight_info_dict[name] = {'weight': w,
                                      'w_sorted_abs': w_sorted_abs,
                                      'zero_percent': zero_percent,
                                      'w_square_sorted': w_square_sorted,
                                      'energy_cummulated': energy_cummulated,
                                      'hw_aware_sparsity': hw_aware_sparsity}

    def _get_target_sparsity(self, name, m):
        target_sparsity = 1
        if not (isinstance(m, SUPPORT_CONV)):  # This target sparsity analysis is only for Convolution  # noqa E501
            return target_sparsity
        groups = m.groups
        channels_out, channels_in = int(m.out_channels / groups), int(m.in_channels / groups)  # noqa E501
        m.kernel_size = convert_pair(m.kernel_size)
        m.stride = convert_pair(m.stride)
        m.dilation = convert_pair(m.dilation)
        (kernel_h, kernel_w) = m.kernel_size
        (stride_h, stride_w) = m.stride
        is_dw = (groups == channels_in)
        (dilations_h, dilations_w) = m.dilation
        dilated_kernel_h, dilated_kernel_w = get_dilated_kernel_size(kernel_h, kernel_w, dilations_h, dilations_w)  # noqa E501
        min_splited_h, min_splited_w = get_min_split_kernel_size(dilated_kernel_h,  # noqa E501
                                                                 dilated_kernel_w,  # noqa E501
                                                                 dilations_h,
                                                                 dilations_w)
        kernel_dec_h = stride_h
        kernel_dec_w = stride_w

        # cv_chip = 'cv22'
        cv_chip = self.prun_hw_aware
        agg_test = [8, 8, 8]
        if self.prun_hw_aware_en:
            target_sparsity = 1 - compute_target_density(cv_chip, agg_test,
                                                         is_dw, channels_out,
                                                         channels_in,
                                                         min_splited_h,
                                                         min_splited_w,
                                                         kernel_dec_h,
                                                         kernel_dec_w)
        return target_sparsity

    def _sparse_ratio_decision(self, energy, stride_group, max_layer_group,
                               layer_name, ctrl_bits, is_dw, prun_algo,
                               prun_algo_tuning,
                               bin_core_sparse_ratio_decision,
                               weight_info_dict):
        # based on energy and layer group to derive final energy for each layer
        # after getting final energy, each layer can get pruning ratio
        # return pruning ratio for each layer
        sparse_ratio = 0.0
        energy_th = energy
        layer_group = max_layer_group - stride_group  # noqa: F841
        weights = weight_info_dict[layer_name]['weight']
        MAX_SPARSE = 0.99
        if weights.dim() == 4:  # convolution
            scale_dw = 2 if is_dw else 1  # noqa: F841
            if prun_algo == 0:
                # this case consider energy_th as pruning ratio directly, instead of energy  # noqa E501
                energy_th = energy_th
            else:
                argc_list = '{} {} {} {} {} {} {}'.format(prun_algo, prun_algo_tuning, energy_th,  # noqa E501
                                                          stride_group, max_layer_group, ctrl_bits, is_dw)  # noqa E501
                energy_th = float(subprocess.Popen('{} {}'.format(bin_core_sparse_ratio_decision, argc_list),  # noqa E501
                                                   shell=True, stdout=subprocess.PIPE).communicate()[0])  # noqa E501
            out_c, in_c, w, h = weights.shape
        elif weights.dim() == 2:  # fully connected
            out_c, in_c = weights.shape

        if prun_algo > 0:
            # use energy as criterion
            energy_cummulated = weight_info_dict[layer_name]['energy_cummulated']  # noqa E501
            len_w = len(energy_cummulated)
            max_index = min(int(len_w * 0.99) + 1, len_w - 1)
            if(energy_cummulated[max_index] * 100.0 < energy_th):
                index = max_index  # Avoid 100% sparsity
            elif (energy_cummulated[0] * 100.0 >= energy_th):
                index = 0
            else:
                index = self.binary_search(energy_cummulated, energy_th)
            # calculate sparse ratio
            sparse_ratio = int(0.5 + index / float(len_w) * 100.0) / 100.0
        else:
            sparse_ratio = energy_th
        """
            use 1.0 will let some network directly die
            so avoid from using 1.0 for pruning ratio
            so add maximum value MAX_SPARSE
        """
        sparse_ratio = min(sparse_ratio, MAX_SPARSE)
        return sparse_ratio

    def binary_search(self, energy_cummulated, desired_per):
        len_w = len(energy_cummulated)
        left = 0
        right = len_w - 1
        while right > left:
            idx = int((right + left) / 2.0)
            if(energy_cummulated[idx] * 100.0 > desired_per):
                if(energy_cummulated[idx - 1] * 100.0 <= desired_per):
                    break

                right = min(idx - 1, len_w - 1)

            else:
                if(energy_cummulated[idx + 1] * 100.0 > desired_per):
                    if(idx != len_w - 1):
                        idx = idx + 1
                    break

                left = max(idx + 1, 0)

        return idx

    def compute_sparse_rate(self, model):
        prune_step = len([i for i in self.iters if i <= self.iteration]) - 1
        sparse_rate = self.sparse_table[prune_step]
        return sparse_rate
