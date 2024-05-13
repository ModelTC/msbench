from torch import Tensor
from torch import nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sconfig=None,
        device=None,
        dtype=None
    ):
        super(Linear, self).__init__(in_features, out_features, False)
        self.sconfig = sconfig
        self.weight_fake_sparse = self.sconfig.weight()

    def forward(self, input: Tensor) -> Tensor:
        sparse_weight = self.weight_fake_sparse(self.weight)
        return F.linear(input, sparse_weight, self.bias)

    @classmethod
    def from_dense(cls, mod):
        assert hasattr(mod, 'sconfig'), 'Input float module must have sconfig defined'
        assert mod.sconfig, 'Input float module must have a valid sconfig'
        sconfig = mod.sconfig
        sparse_linear = cls(mod.in_features, mod.out_features, mod.bias is not None, sconfig)
        sparse_linear.weight = mod.weight
        sparse_linear.bias = mod.bias
        return sparse_linear
