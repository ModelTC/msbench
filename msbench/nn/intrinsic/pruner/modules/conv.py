from torch import Tensor
from torch import nn
from torch.nn.common_types import _size_2_t


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',  # TODO: refine this type
        sconfig=None
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.sconfig = sconfig
        self.weight_fake_sparse = self.sconfig.weight()

    def forward(self, input: Tensor) -> Tensor:
        sparse_weight = self.weight_fake_sparse(self.weight)
        return self._conv_forward(input, sparse_weight, self.bias)

    @classmethod
    def from_dense(cls, mod):
        assert hasattr(mod, 'sconfig'), 'Input float module must have sconfig defined'
        assert mod.sconfig, 'Input float module must have a valid sconfig'
        sconfig = mod.sconfig
        sparse_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                          mod.stride, mod.padding, mod.dilation,
                          mod.groups, mod.bias is not None,
                          mod.padding_mode,
                          sconfig)
        sparse_conv.weight = mod.weight
        sparse_conv.bias = mod.bias
        return sparse_conv
