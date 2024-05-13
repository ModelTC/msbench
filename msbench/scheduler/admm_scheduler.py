import torch
from msbench.sparsity_mappings import SUPPORT_CONV, SUPPORT_LINEAR
from msbench.scheduler.base_scheduler import BaseScheduler
from msbench.utils.state import disable_sparsification, enable_sparsification
from msbench.utils.logger import logger

_SUPPORT_MODULE_TYPES = (torch.nn.Conv2d, torch.nn.Linear)


class ADMMScheduler(BaseScheduler):
    def __init__(self, custom_config_dict, no_prune_keyword='',
                 no_prune_layer=''):
        super().__init__(custom_config_dict, no_prune_keyword, no_prune_layer)
        print("self.custom_config_dict : ", self.custom_config_dict)
        self.target_sparsity = self.custom_config_dict['sparsity']['target_sparsity']
        self.uniform = self.custom_config_dict['sparsity']['uniform']
        self.structured = self.custom_config_dict['MaskGeneratorConfig']['kwargs']['structured']
        self.func = self.custom_config_dict['MaskGeneratorConfig']['kwargs']['func']

    def before_run(self, model, iters_per_epoch, max_iters):
        self.iters_per_epoch = iters_per_epoch
        self.max_iters = max_iters
        self.max_epochs = int(max_iters / iters_per_epoch)
        self.admm_epochs = int(self.max_epochs * 1 / 3)
        self.admm_iters = self.admm_epochs * self.iters_per_epoch
        self.Z = {}
        self.U = {}
        for name, m in model.named_modules():
            if isinstance(m, _SUPPORT_MODULE_TYPES):
                self.Z[name] = {'weight': m.weight.data.clone()}
                self.U[name] = {'weight': torch.zeros_like(m.weight.data)}
        disable_sparsification(model)

    def prune_model(self, model, cur_iter):
        res = {}
        if cur_iter < self.admm_iters:
            if cur_iter % self.iters_per_epoch == 0:
                for name, m in model.named_modules():
                    if isinstance(m, _SUPPORT_MODULE_TYPES):
                        self.Z[name]['weight'] = m.weight.data + self.U[name]['weight']
                
                masks = self.get_masks()
                
                for name, m in model.named_modules():
                    if isinstance(m, _SUPPORT_MODULE_TYPES):
                        self.Z[name]['weight'] *= masks[name]

                for name, m in model.named_modules():
                    if isinstance(m, _SUPPORT_MODULE_TYPES):
                        self.U[name]['weight'] = m.weight.data + self.U[name]['weight'] - self.Z[name]['weight']

                penalty = 0
                rho = 1e-4
                for name, m in model.named_modules():
                    if isinstance(m, _SUPPORT_MODULE_TYPES):
                        penalty += (rho / 2) * torch.sqrt(torch.norm(m.weight.data - self.Z[name]['weight'] + self.U[name]['weight']))
                
                res = {'loss': penalty}

        elif cur_iter == self.admm_iters:
            sparsities = self.get_sparsity(model)
            self.update_sparsity_per_layer_from_sparsities(model, sparsities)
            enable_sparsification(model)

        return res


    def get_unstructured_nonuniform_mask(self, Z, target_sparsity, func='L2Normalized'):
        all_weights = []
        for name, module in Z.items():
            if func == 'Magnitude':
                all_weights.append(torch.flatten(module['weight']))
            elif func == 'L1Normalized':
                all_weights.append(torch.flatten(module['weight']) / torch.norm(module['weight'], p=1))
            elif func == 'L2Normalized':
                all_weights.append(torch.flatten(module['weight']) / torch.norm(module['weight'], p=2))
            else:
                print("Not support func.")
        all_weights = torch.cat(all_weights)
        all_weights = torch.absolute(all_weights)
        all_weights, _ = all_weights.sort()
        sparsity_threshold = all_weights[int(float(target_sparsity) * len(all_weights))]
        nonzeros_masks = {}
        for name, module in Z.items():
            if func == 'Magnitude':
                mask = torch.absolute(module['weight']) > sparsity_threshold
            elif func == 'L1Normalized':
                mask = torch.absolute(module['weight'] / torch.norm(module['weight'], p=1)) > sparsity_threshold
            elif func == 'L2Normalized':
                mask = torch.absolute(module['weight'] / torch.norm(module['weight'], p=2)) > sparsity_threshold
            else:
                print("Not support func.")
            nonzeros_masks[name] = mask
        return nonzeros_masks

    def get_unstructured_nonuniform_sparsity(self, model, target_sparsity, func='L2Normalized'):
        all_weights = []
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                if func == 'Magnitude':
                    all_weights.append(torch.flatten(module.weight))
                elif func == 'L1Normalized':
                    all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=1))
                elif func == 'L2Normalized':
                    all_weights.append(torch.flatten(module.weight) / torch.norm(module.weight, p=2))
                else:
                    print("Not support func.")
        all_weights = torch.cat(all_weights)
        all_weights = torch.absolute(all_weights)
        all_weights, _ = all_weights.sort()
        sparsity_threshold = all_weights[int(float(target_sparsity) * len(all_weights))]
        sparsities = {}
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                if func == 'Magnitude':
                    mask = torch.absolute(module.weight) > sparsity_threshold
                elif func == 'L1Normalized':
                    mask = torch.absolute(module.weight / torch.norm(module.weight, p=1)) > sparsity_threshold
                elif func == 'L2Normalized':
                    mask = torch.absolute(module.weight / torch.norm(module.weight, p=2)) > sparsity_threshold
                else:
                    print("Not support func.")
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
                sparsities[name] = sparsity
        return sparsities

    def get_unstructured_uniform_mask(self, Z, target_sparsity):
        nonzeros_masks = {}
        for name, module in Z.items():
            weights = torch.absolute(torch.flatten(module['weight']))
            weights, _ = weights.sort()
            sparsity_threshold = weights[int(float(target_sparsity) * len(weights))]
            mask = torch.absolute(module['weight']) > sparsity_threshold
            nonzeros_masks[name] = mask
        return nonzeros_masks

    def get_unstructured_uniform_sparsity(self, model, target_sparsity):
        sparsities = {}
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                sparsities[name] = target_sparsity
        return sparsities

    def get_structured_nonuniform_mask(self, Z, target_sparsity, func='p1'):
        all_weights = []
        for name, module in Z.items():
            if func == 'p1':
                weights = torch.absolute(torch.norm(module['weight'].reshape(module['weight'].shape[0], -1), p=1, dim=-1)) / (module['weight'].numel() / module['weight'].shape[0])
                all_weights += [weights] * int(module['weight'].numel() / module['weight'].shape[0])
            elif func == 'p2':
                weights = torch.absolute(torch.norm(module['weight'].reshape(module['weight'].shape[0], -1), p=2, dim=-1)) / (module['weight'].numel() / module['weight'].shape[0])
                all_weights += [weights] * int(module['weight'].numel() / module['weight'].shape[0])
            else:
                print("Not support func.")
        all_weights = torch.cat(all_weights)
        # all_weights = torch.absolute(all_weights)
        all_weights, _ = all_weights.sort()
        sparsity_threshold = all_weights[int(float(target_sparsity) * len(all_weights))]
        nonzeros_masks = {}
        for name, module in Z.items():
            if func == 'p1':
                mask = weights > sparsity_threshold
            elif func == 'p2':
                mask = weights > sparsity_threshold
            else:
                print("Not support func.")
            mask = mask.view(-1, *(1,) * (len(module['weight'].shape) - 1)).expand(module['weight'].shape)
            nonzeros_masks[name] = mask
        return nonzeros_masks

    def get_structured_nonuniform_sparsity(self, model, target_sparsity, func='p1'):
        all_weights = []
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                if func == 'p1':
                    weights = torch.absolute(torch.norm(module.weight.reshape(module.weight.shape[0], -1), p=1, dim=-1)) / (module.weight.numel() / module.weight.shape[0])
                    all_weights += [weights] * int(module.weight.numel() / module.weight.shape[0])
                elif func == 'p2':
                    weights = torch.absolute(torch.norm(module.weight.reshape(module.weight.shape[0], -1), p=2, dim=-1)) / (module.weight.numel() / module.weight.shape[0])
                    all_weights += [weights] * int(module.weight.numel() / module.weight.shape[0])
                else:
                    print("Not support func.")
        all_weights = torch.cat(all_weights)
        # all_weights = torch.absolute(all_weights)
        all_weights, _ = all_weights.sort()
        sparsity_threshold = all_weights[int(float(target_sparsity) * len(all_weights))]
        sparsities = {}
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                if func == 'p1':
                    mask = weights > sparsity_threshold
                elif func == 'p2':
                    mask = weights > sparsity_threshold
                else:
                    print("Not support func.")
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
                sparsities[name] = sparsity
        return sparsities

    def get_structured_uniform_mask(self, Z, target_sparsity, func='p1'):
        nonzeros_masks = {}
        for name, module in Z.items():
            if func == 'p1':
                weights = torch.absolute(torch.norm(module['weight'].reshape(module['weight'].shape[0], -1), p=1, dim=-1))
                weights_sorted, _ = weights.sort()
                sparsity_threshold = weights_sorted[int(float(target_sparsity) * len(weights_sorted))]
                mask = weights > sparsity_threshold
            elif func == 'p2':
                weights = torch.absolute(torch.norm(module['weight'].reshape(module['weight'].shape[0], -1), p=2, dim=-1))
                weights_sorted, _ = weights.sort()
                sparsity_threshold = weights_sorted[int(float(target_sparsity) * len(weights_sorted))]
                mask = weights > sparsity_threshold
            else:
                print("Not support func.")
            mask = mask.view(-1, *(1,) * (len(module['weight'].shape) - 1)).expand(module['weight'].shape)
            nonzeros_masks[name] = mask
        return nonzeros_masks

    def get_structured_uniform_sparsity(self, model, target_sparsity, func='p1'):
        sparsities = {}
        for name, module in model.named_modules():
            if isinstance(module, _SUPPORT_MODULE_TYPES):
                if func == 'p1':
                    weights = torch.absolute(torch.norm(module.weight.reshape(module.weight.shape[0], -1), p=1, dim=-1))
                    weights_sorted, _ = weights.sort()
                    sparsity_threshold = weights_sorted[int(float(target_sparsity) * len(weights_sorted))]
                    mask = weights > sparsity_threshold
                elif func == 'p2':
                    weights = torch.absolute(torch.norm(module.weight.reshape(module.weight.shape[0], -1), p=2, dim=-1))
                    weights_sorted, _ = weights.sort()
                    sparsity_threshold = weights_sorted[int(float(target_sparsity) * len(weights_sorted))]
                    mask = weights > sparsity_threshold
                else:
                    print("Not support func.")
                sparsity = 1 - float(torch.count_nonzero(mask)) / module.weight.numel()
                sparsities[name] = sparsity
        return sparsities

    def get_masks(self):
        if self.structured:
            if self.uniform:
                nonzeros_masks = self.get_structured_uniform_mask(self.Z, target_sparsity=self.target_sparsity, func=self.func)
            else:
                nonzeros_masks = self.get_structured_nonuniform_mask(self.Z, target_sparsity=self.target_sparsity, func=self.func)
        else:
            if self.uniform:
                nonzeros_masks = self.get_unstructured_uniform_mask(self.Z, target_sparsity=self.target_sparsity)
            else:
                nonzeros_masks = self.get_unstructured_nonuniform_mask(self.Z, target_sparsity=self.target_sparsity, func=self.func)
        return nonzeros_masks

    def get_sparsity(self, model):
        if self.structured:
            if self.uniform:
                nonzeros_masks = self.get_structured_uniform_sparsity(model, target_sparsity=self.target_sparsity, func=self.func)
            else:
                nonzeros_masks = self.get_structured_nonuniform_sparsity(model, target_sparsity=self.target_sparsity, func=self.func)
        else:
            if self.uniform:
                nonzeros_masks = self.get_unstructured_uniform_sparsity(model, target_sparsity=self.target_sparsity)
            else:
                nonzeros_masks = self.get_unstructured_nonuniform_sparsity(model, target_sparsity=self.target_sparsity, func=self.func)
        return nonzeros_masks

    def update_sparsity_per_layer_from_sparsities(self, model, sparsities):
        zero_nums = 0
        total_nums = 0
        for name, m in model.named_modules():
            if isinstance(m, _SUPPORT_MODULE_TYPES):
                final_sparsity = sparsities[name]
                m.weight_fake_sparse.mask_generator.sparsity = final_sparsity
                zero_nums += final_sparsity * m.weight.numel()
                total_nums += m.weight.numel()
        logger.info("After compute, real sparsity = {}".format(zero_nums / total_nums))


