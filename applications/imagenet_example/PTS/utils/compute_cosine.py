import torch
import torch.nn as nn
from torchvision.models import resnet18
from msbench.utils.hook import DataSaverHook, StopForwardException
import copy
import numpy as np


def save_inp_oup_data(model, inp_module, oup_module, cali_data: list, store_inp=True, store_oup=True,
                      keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.
    :param cali_data: calibration data set
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    if store_inp:
        assert inp_module is not None
        inp_saver = DataSaverHook(store_input=store_inp, store_output=False, stop_forward=(not store_oup))
        inp_handle = inp_module.register_forward_hook(inp_saver)
    if store_oup:
        assert oup_module is not None
        oup_saver = DataSaverHook(store_input=False, store_output=store_oup, stop_forward=True)
        oup_handle = oup_module.register_forward_hook(oup_saver)
    cached = ([], [])
    with torch.no_grad():
        for batch in cali_data:
            try:
                _ = model(batch.to(device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append([inp.detach() for inp in inp_saver.input_store])
                else:
                    cached[0].append([inp.detach().cpu() for inp in inp_saver.input_store])  # tuple/list one
            if store_oup:
                if keep_gpu:
                    cached[1].append(oup_saver.output_store.detach())
                else:
                    cached[1].append(oup_saver.output_store.detach().cpu())
    if store_inp:
        inp_handle.remove()
    if store_oup:
        oup_handle.remove()
    torch.cuda.empty_cache()
    return cached

def get_outputs_perlayer(model, inputs_list):
    model.eval()
    oup_handle_list = []
    oup_saver_list = []
    name_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            oup_saver = DataSaverHook(store_input=False, store_output=True, stop_forward=False)
            oup_handle = module.register_forward_hook(oup_saver)
            oup_saver_list.append(oup_saver)
            oup_handle_list.append(oup_handle)
            name_list.append(name)
    device = next(model.parameters()).device
    cached = {}
    for inputs in inputs_list:
        with torch.no_grad():
            try:
                _ = model(inputs.to(device))
            except StopForwardException:
                pass
            for name, oup_saver in zip(name_list, oup_saver_list):
                if name not in cached:
                    cached[name] = oup_saver.output_store.detach().cpu()
                else:
                    cached[name] = torch.cat((cached[name],oup_saver.output_store.detach().cpu()), dim=0)
    for oup_handle in oup_handle_list:
        oup_handle.remove()
    torch.cuda.empty_cache()
    for key, value in cached.items():
        print(key, value.shape)
    return cached

def cos_similarity(ta, tb):
    assert ta.shape == tb.shape
    if np.sum(ta * tb) == 0:
        return 0.
    return np.sum(ta * tb) / np.sqrt(np.square(ta).sum()) \
        / np.sqrt(np.square(tb).sum())

def get_cos_similarity(demse_model, sparse_model, inputs_list):
    dense_cached = get_outputs_perlayer(demse_model, inputs_list)
    sparse_cached = get_outputs_perlayer(sparse_model, inputs_list)
    cosine_dict = {}
    for dense_key, sparse_key in zip(dense_cached.keys(), sparse_cached.keys()):
        # cosine = torch.cosine_similarity(dense_cached[dense_key], sparse_cached[sparse_key])
        cosine = cos_similarity(dense_cached[dense_key].numpy(), sparse_cached[sparse_key].numpy())
        print("layer name: {0}, cosine: {1}".format(dense_key, cosine))
        cosine_dict[dense_key] = cosine
    return cosine_dict

def test1():
    model = resnet18(pretrained=True)
    dummy_inputs = torch.randn(1,3,224,224)
    dense_cached = get_outputs_perlayer(model, [dummy_inputs, dummy_inputs])

    sparse_model = copy.deepcopy(model)
    sparse_cached = get_outputs_perlayer(model, [dummy_inputs, dummy_inputs])

    for dense_key, sparse_key in zip(dense_cached.keys(), sparse_cached.keys()):
        # cosine = torch.cosine_similarity(dense_cached[dense_key], sparse_cached[sparse_key])
        cosine = cos_similarity(dense_cached[dense_key].numpy(), sparse_cached[sparse_key].numpy())
        print("layer name: {0}, cosine: {1}".format(dense_key, cosine))


def test2():
    from msbench.scheduler import build_sparse_scheduler
    sparse_cfg = {
        'cali_batchsize': 2,
        'reconstruction': {
            'pattern': 'block',
            'max_count': 10,
            'keep_gpu': True,
            'weight_lr': 0.00001},
        'mask_generator': {
            'type': 'NormalMaskGenerator',
            'kwargs':{
                'sparsity': 0.9
            }
        },
        'fake_sparse':{
            'type': 'FakeSparse'
        },
        'scheduler':{
            'type': 'BaseScheduler'
        }
    }

    model = resnet18(pretrained=True)
    dummy_inputs = torch.randn(1,3,224,224)
    dense_cached = get_outputs_perlayer(model, [dummy_inputs])
    sparse_model = copy.deepcopy(model)
    sparse_scheduler = build_sparse_scheduler(sparse_cfg)
    sparse_model = sparse_scheduler.prepare_sparse_model(sparse_model)
    sparse_cached = get_outputs_perlayer(sparse_model, [dummy_inputs])


    for dense_key, sparse_key in zip(dense_cached.keys(), sparse_cached.keys()):
        # cosine = torch.cosine_similarity(dense_cached[dense_key], sparse_cached[sparse_key])
        cosine = cos_similarity(dense_cached[dense_key].numpy(), sparse_cached[sparse_key].numpy())
        print("layer name: {0}, cosine: {1}".format(sparse_key, cosine))

if  __name__ == '__main__':
    test2()
