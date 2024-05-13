from torch.fx.graph_module import GraphModule
import copy
from msbench.sparsity_mappings import DEFAULT_REFERENCE_STATIC_SPARSE_MODULE_MAPPINGS


class ModelSparser(object):
    def __init__(self, config_dict):
        self.sparse_mapping = config_dict.get('sparse_mapping', DEFAULT_REFERENCE_STATIC_SPARSE_MODULE_MAPPINGS)

    def prepare(self, model: GraphModule, sconfig):
        self.propagate_sconfig_(model, sconfig)
        graph = self._convert(model, self.sparse_mapping, True)
        return graph

    def propagate_sconfig_(self, module, sconfig):
        module.sconfig = sconfig
        for name, child in module.named_children():
            self.propagate_sconfig_(child, sconfig)
        return module

    def _convert(self, module, mapping=None, inplace=False):
        if not inplace:
            module = copy.deepcopy(module)
        reassign = {}
        for name, mod in module.named_children():
            self._convert(mod, mapping, True)
            reassign[name] = self.swap_module(mod, mapping)

        for key, value in reassign.items():
            module._modules[key] = value

        return module

    def swap_module(self, mod, mapping):
        new_mod = mod
        if hasattr(mod, 'sconfig') and mod.sconfig is not None:
            swapped = False
            if type(mod) in mapping:
                new_mod = mapping[type(mod)].from_dense(mod)
                swapped = True

            if swapped:
                for pre_hook_fn in mod._forward_pre_hooks.values():
                    new_mod.register_forward_pre_hook(pre_hook_fn)
                for hook_fn in mod._forward_hooks.values():
                    new_mod.register_forward_hook(hook_fn)

                devices = self.get_unique_devices_(mod)
                assert len(devices) <= 1, (
                    "swap_module only works with cpu or single-device CUDA modules, "
                    "but got devices {}".format(devices)
                )
                device = next(iter(devices)) if len(devices) > 0 else None
                if device:
                    new_mod.to(device)
        return new_mod

    def get_unique_devices_(self, module):
        return {p.device for p in module.parameters()} | \
            {p.device for p in module.buffers()}
