
from typing import List, Optional, Union
from abc import abstractmethod
import torch
from torch import Tensor
import torch.nn.functional as F

class MaskGeneratorBase:
    def __init__(self):
        self.sparsity = None

    @abstractmethod
    def generate_mask(self, **kwargs):
        pass

    def _expand_mask(self, mask):
        pass

    def _compress_mask(self, mask):
        pass


# https://arxiv.org/abs/1811.00250
class FPGMMaskGenerator(MaskGeneratorBase):

    def __init__(self, structured, func=None):
        super().__init__()
        self.structured = structured
        assert self.structured is True
        self.func = func

    def calculate_distance_matrix(self, weight):
        flattened_kernels = weight.view(weight.size(0), -1)
        if self.func == 'l1':
            distance_matrix = torch.cdist(flattened_kernels, flattened_kernels, p=1)
        elif self.func == 'l2':
            distance_matrix = torch.cdist(flattened_kernels, flattened_kernels, p=2)
        elif self.func == 'cos':
            distance_matrix = 1 - F.cosine_similarity(flattened_kernels[:, None], flattened_kernels, dim=2)
        
        return distance_matrix

    def generate_mask(self, weight):
        num_kernels_to_keep = int(weight.shape[0] * (1 - self.sparsity))
        distance_matrix = self.calculate_distance_matrix(weight)
        mean_distances = distance_matrix.mean(dim=1)
        sorted_indices = torch.argsort(mean_distances, descending=True)
        mask = torch.zeros_like(weight)
        important_indices = sorted_indices[:num_kernels_to_keep]
        mask[important_indices] = 1

        return mask


# https://www.sciencedirect.com/science/article/abs/pii/S0031320324002395
class NuclearMaskGenerator(MaskGeneratorBase):

    def __init__(self, structured, func=None):
        super().__init__()
        self.structured = structured
        assert self.structured is True

    def generate_mask(self, weight):
        
        flattened_weights = weight.view(weight.size(0), -1)
        original_norm = torch.linalg.matrix_norm(flattened_weights, ord='nuc')
        impacts = []
        for i in range(flattened_weights.size(0)):
            temp_weights = torch.cat([flattened_weights[:i], flattened_weights[i+1:]], dim=0)
            reduced_norm = torch.linalg.matrix_norm(temp_weights, ord='nuc')
            impact = original_norm - reduced_norm
            impacts.append(impact.item())

        impacts = torch.tensor(impacts)
        sorted_indices = torch.argsort(impacts, descending=True)
        mask = torch.zeros_like(weight)

        num_channels_to_keep = int(weight.shape[0] * (1 - self.sparsity))
        indices_to_keep = sorted_indices[:num_channels_to_keep]
        mask[indices_to_keep] = 1

        return mask


class NormalMaskGenerator(MaskGeneratorBase):
    r"""
    This genrrator simply pruned the weight with smaller metrics in layer level.
    """

    def __init__(self, structured, func=None):
        super().__init__()
        self.structured = structured
        if self.structured:
            if func == 'p1':
                self.p = 1
            elif func == 'p2':
                self.p = 2

    def generate_mask(self, metrics: Tensor):
        if self.structured:
            revised_metrics = torch.norm(metrics.reshape(metrics.shape[0], -1), p=self.p, dim=-1)
        else:
            revised_metrics = metrics.abs()

        prune_num = revised_metrics.numel() - int((1-self.sparsity) * revised_metrics.numel())
        if prune_num == 0:
            threshold = revised_metrics.min() - 1
        else:
            threshold = torch.topk(revised_metrics.view(-1), prune_num, largest=False)[0].max()
        mask = torch.gt(revised_metrics, threshold).type_as(revised_metrics)
    
        if self.structured:
            mask = mask.view(-1, *(1,) * (len(metrics.shape) - 1)).expand(metrics.shape)
        return mask


class NMMaskGenerator(MaskGeneratorBase):
    r"""
    This genrrator simply pruned the weight with smaller metrics in layer level.
    """

    def __init__(self):
        super().__init__()

    def generate_mask(self, metrics: Tensor):
        mask = self.create_nm_mask(metrics, 2, 4)
        return mask

    def create_nm_mask(self, weight, N=2, M=4):
        if weight.shape[1] % M != 0:
            mask = torch.ones(weight.shape, device=weight.device, requires_grad=False)
            return mask
        if len(weight.shape) == 4:
            weight_temp = weight.detach().abs().permute(0, 2, 3, 1).reshape(-1, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M - N)]
            mask = torch.ones(weight_temp.shape, device=weight_temp.device)
            mask = mask.scatter_(dim=1, index=index, value=0).reshape((weight.shape[0], weight.shape[2], weight.shape[3], weight.shape[1]))
            mask = mask.permute(0, 3, 1, 2)
        elif len(weight.shape) == 2:
            weight_temp = weight.detach().abs().reshape(-1, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M - N)]
            mask = torch.ones(weight_temp.shape, device=weight_temp.device)
            mask = mask.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        else:
            raise Exception(f"Not support weight shape: {weight.shape}.") 
        return mask


class ProbMaskGenerator(MaskGeneratorBase):
    def __init__(self, tau: float = 3.0) -> None:
        super().__init__()
        self.eps = 1e-20
        self.tau = tau

    def enable_fix_subnet(self, scores) -> None:
        self.mask = (torch.rand_like(scores) < scores).float()

    def set_tau(self, tau):
        self.tau = tau

    def generate_mask(self, scores, fix_subnet):
        if not fix_subnet:
            torch.set_grad_enabled(True)
            uniform0 = torch.rand_like(scores)
            uniform1 = torch.rand_like(scores)
            noise = -torch.log(torch.log(uniform0 + self.eps) / torch.log(uniform1 + self.eps) + self.eps)
            self.mask = torch.sigmoid((torch.log(scores + self.eps) - torch.log(1.0 - scores + self.eps) + noise) * self.tau)
        return self.mask


class STRMaskGenerator(MaskGeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = torch.relu
        self.f = torch.sigmoid

    def generate_sparse_weight(self, weight, scores):
        return torch.sign(weight) * self.activation(torch.abs(weight) - self.f(scores))

    def generate_mask(self, weight, scores):
        return (torch.abs(weight) > self.f(scores)).float()
