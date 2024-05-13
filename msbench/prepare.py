import torch
from torch.fx import Tracer
import torch.nn as nn
from msbench.custom_sparser import ModelSparser
from torch.fx.graph_module import GraphModule
from msbench.fake_sparse import DefaultFakeSparse, ProbMaskFakeSparse, STRFakeSparse
from msbench.mask_generator import NormalMaskGenerator, ProbMaskGenerator, STRMaskGenerator, NuclearMaskGenerator, FPGMMaskGenerator, NMMaskGenerator
from msbench.utils.fuse_bn import fuse
from collections import namedtuple

MaskGeneratorTable = {
    'NormalMaskGenerator': NormalMaskGenerator,
    'ProbMaskGenerator': ProbMaskGenerator,
    'STRMaskGenerator': STRMaskGenerator,
    'NuclearMaskGenerator': NuclearMaskGenerator,
    'FPGMMaskGenerator': FPGMMaskGenerator,
    'NMMaskGenerator': NMMaskGenerator,
}

FakeSparseTable = {
    'DefaultFakeSparse': DefaultFakeSparse,
    'ProbMaskFakeSparse': ProbMaskFakeSparse,
    'STRFakeSparse': STRFakeSparse
}


class CustomedTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is
    equivalent to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module.
            For example, if you have a module hierarchy where
            submodule ``foo`` contains submodule ``bar``,
            which contains submodule ``baz``, that module will
            appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


def prepare(model, prepare_custom_config_dict):
    # Symbolic trace
    customed_leaf_module = prepare_custom_config_dict.get('leaf_module', [])
    tracer = CustomedTracer(customed_leaf_module=tuple(customed_leaf_module))
    graph = tracer.trace(model)
    name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
    graph = GraphModule(tracer.root, graph, name)
    if prepare_custom_config_dict['fuse_bn']:
        graph.eval()
        graph = fuse(graph)
        graph.train()

    MaskGeneratorConfig = prepare_custom_config_dict['MaskGeneratorConfig']
    FakeSparseConfig = prepare_custom_config_dict['FakeSparseConfig']
    if MaskGeneratorConfig.get('kwargs', None) is None:
        MaskGeneratorConfig['kwargs'] = {}
    if FakeSparseConfig.get('kwargs', None) is None:
        FakeSparseConfig['kwargs'] = {}
    kwargs = {
        "FakeSparseConfig": FakeSparseConfig['kwargs'],
        "MaskGeneratorConfig": MaskGeneratorConfig['kwargs']
    }
    FakeSparseMask = FakeSparseTable[FakeSparseConfig['type']].with_args(generator=MaskGeneratorTable[MaskGeneratorConfig['type']], **kwargs)
    sconfig = namedtuple('SConfig', ['weight'])(FakeSparseMask)

    sparser = ModelSparser(prepare_custom_config_dict)
    model_prepared = sparser.prepare(graph, sconfig)

    return model_prepared
