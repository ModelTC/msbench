"""
This module implements modules which are used to perform fake sparse during
sparse training.
"""

import torch
import torch.nn as nn
from torch.nn import Module
from abc import ABC, abstractmethod
from functools import partial
from copy import deepcopy
from msbench.mask_generator import (
    NormalMaskGenerator,
    ProbMaskGenerator,
    STRMaskGenerator
)


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.
    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.
    Example::
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


class FakeSparseBase(ABC, Module):
    r""" Base fake sparse module
    Any fake sparse implementation should derive from this class.
    Concrete fake sparse module should follow the same API. In forward, they will
    fake sparse the input. They should also provide a `generate_mask` function
    that computes the mask of input.
    """

    fake_sparse_enabled: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        # fake_sparse_enabled are buffers to support their replication in DDP.
        # Data type is uint8 because NCCL does not support bool tensors.
        self.register_buffer('fake_sparse_enabled', torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def generate_mask(self, **kwargs):
        pass

    @torch.jit.export
    def enable_fake_sparse(self, enabled: bool = True) -> None:
        self.fake_sparse_enabled[0] = 1 if enabled else 0

    @torch.jit.export
    def disable_fake_sparse(self):
        self.enable_fake_sparse(False)

    with_args = classmethod(_with_args)


class DefaultFakeSparse(FakeSparseBase):
    r""" Simulate the sparse operations in training time.
    The output of this module is given by::
        x_out = x * mask
    * :attr:`mask` defines the mask used for weight.
    * :attr:`fake_sparse_enabled` controls the application of fake sparse on tensors, note that
      statistics can still be updated.
    Args:
        sparsity (float): sparsity rate in [0, 1.0).
    Attributes:
        generator (Module): User provided module that analysis statistics on the input tensor
        provides the mask of the input tensor.
    """

    mask: torch.Tensor

    def __init__(self, generator=NormalMaskGenerator, **kwargs) -> None:
        super().__init__()
        FakeSparseConfig = kwargs.get('FakeSparseConfig', {})
        self.mask_is_static = FakeSparseConfig.get('mask_is_static', False)
        self.weight_correction = FakeSparseConfig.get('weight_correction', False)
        self.register_buffer('mask_tensor', None)
        self.mask_generator = generator(**kwargs.get('MaskGeneratorConfig', {}))

    @torch.jit.export
    @torch.no_grad()
    def generate_mask(self, metrics):
        if self.mask_is_static:
            return self.mask_tensor
        return self.mask_generator.generate_mask(metrics)

    @torch.no_grad()
    def before_run(self, metrics):
        if self.mask_is_static:
            self.mask_tensor = self.mask_generator.generate_mask(metrics)
        if self.weight_correction:
            mask = self.mask_generator.generate_mask(metrics) if self.mask_tensor is None else self.mask_tensor
            sparse_weights = deepcopy(metrics)
            sparse_weights = sparse_weights * mask
            axis = tuple(range(1, len(metrics.shape)))
            broadcast_axes = (Ellipsis,) + (None,) * (len(metrics.shape) - 1)
            variance_per_channel_shift = torch.std(metrics, axis=axis) / (
                torch.std(sparse_weights, axis=axis) + 10**(-9)
            )
            variance_per_channel_shift = variance_per_channel_shift[broadcast_axes]
            mean_per_channel_shift = torch.mean(metrics, axis=axis) - torch.mean(
                sparse_weights * variance_per_channel_shift, axis=axis
            )
            mean_per_channel_shift = mean_per_channel_shift[broadcast_axes]
            metrics.set_(
                metrics * variance_per_channel_shift + mean_per_channel_shift
            )

    def forward(self, X):
        if self.fake_sparse_enabled[0] == 1:
            mask = self.generate_mask(X)
            X = X * mask
        return X


class ProbMaskFakeSparse(FakeSparseBase):
    r""" Simulate the sparse operations in training time.
    The output of this module is given by::
        x_out = x * mask
    * :attr:`mask` defines the mask used for weight.
    * :attr:`fake_sparse_enabled` controls the application of fake sparse on tensors, note that
      statistics can still be updated.
    Args:
        sparsity (float): sparsity rate in [0, 1.0).
    Attributes:
        generator (Module): User provided module that analysis statistics on the input tensor
        provides the mask of the input tensor.
    """

    mask: torch.Tensor

    def __init__(self, generator=ProbMaskGenerator, score_init_constant=1.0, **generator_kwargs) -> None:
        super().__init__()
        self.mask_generator = generator(**generator_kwargs)
        self.scores = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        self.score_init_constant = score_init_constant
        self.register_buffer('init_score', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('fix_subnet', torch.tensor([0], dtype=torch.uint8))

    @torch.jit.export
    @torch.no_grad()
    def generate_mask(self, metrics):
        return self.mask_generator.generate_mask(metrics, self.fix_subnet[0])

    @property
    def clamped_scores(self):
        return self.scores

    def enable_fix_subnet(self, enabled: bool = False):
        self.fix_subnet[0] = 1 if enabled else 0
        self.mask_generator.enable_fix_subnet(self.scores)

    def forward(self, X):
        if self.init_score[0] == 0:
            self.scores.data = torch.ones_like(X) * self.score_init_constant
            self.init_score[0] = 1
        if self.fake_sparse_enabled[0] == 1:
            mask = self.generate_mask(self.clamped_scores)
            X = X * mask
        return X


class STRFakeSparse(FakeSparseBase):
    mask: torch.Tensor

    def __init__(self, generator=STRMaskGenerator, sInit_value=-200, **generator_kwargs) -> None:
        super().__init__()
        self.mask_generator = generator(**generator_kwargs)
        self.scores = nn.Parameter(torch.ones(1, 1) * sInit_value, requires_grad=True)

    def generate_sparse_weight(self, metrics):
        return self.mask_generator.generate_sparse_weight(metrics, self.scores)

    @torch.jit.export
    @torch.no_grad()
    def generate_mask(self, metrics):
        return self.mask_generator.generate_mask(metrics, self.scores)

    def forward(self, X):
        if self.fake_sparse_enabled[0] == 1:
            X = self.generate_sparse_weight(X)
        return X
