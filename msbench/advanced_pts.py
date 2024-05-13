import torch
from torch.fx import GraphModule, Node
from torch import fx, nn
from torch.nn import Module
import numpy as np
from typing import List
from msbench.utils.logger import logger
from msbench.utils.hook import DataSaverHook, StopForwardException
from msbench.utils import deepcopy_graphmodule
from msbench.utils.state import enable_sparsification, disable_sparsification

USE_LINK = False
USE_DDP = False
__all__ = ['pts_reconstruction']


_SUPPORT_MODULE_TYPES = (torch.nn.Conv2d, torch.nn.Linear)


def before_run(model):
    for name, m in model.named_modules():
        if isinstance(m, _SUPPORT_MODULE_TYPES):
            m.weight_fake_sparse.before_run(m.weight)

def lp_loss(pred, tgt):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).pow(2.0).sum(1).mean()


def save_inp_oup_data(model: GraphModule, inp_module: Module, oup_module: Module, cali_data: list, store_inp=True, store_oup=True,
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


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 subgraph: Module,
                 p: float = 2.):

        self.subgraph = subgraph
        self.p = p
        self.count = 0

    def __call__(self, pred, tgt):
        self.count += 1
        rec_loss = lp_loss(pred, tgt)
        if self.count % 1000 == 0:
            # if link.get_rank() == 0:
            #     logger.info('rec loss:\t{}\tcount={}'.format(float(rec_loss), self.count))
            logger.info('rec loss:\t{}\tcount={}'.format(float(rec_loss), self.count))
        return rec_loss


def _flatten_args(node):
    flattned_args = []
    if isinstance(node, dict):
        for v in node.values():
            flattned_args.extend(_flatten_args(v))
    elif isinstance(node, tuple) or isinstance(node, list):
        for n in node:
            flattned_args.extend(_flatten_args(n))
    else:
        flattned_args.extend([node])
    return flattned_args


def append_extra_inputs(nodes, layer_node_list):
    # there are some nodes in the block which are used but not in the list.
    # e.g. a global dict used in UP or EOD.
    extra_inputs = []
    for node in layer_node_list:
        for arg in _flatten_args(node.args):
            if isinstance(arg, torch.fx.Node):
                if arg not in layer_node_list:
                    extra_inputs.append(arg)
    return extra_inputs


def find_cur_node(layer_node_list):
    print("layer_node_list is ", layer_node_list)
    for node in reversed(layer_node_list):
        if node.target == 'update':
            continue
        if isinstance(node.target, str) and 'const' in node.target:
            continue
        if node.op == 'call_method' or node.op == 'call_function':
            continue
        return node
    raise ValueError('Bad layer node list provided.')


def subgraph_reconstruction(subgraph, cached_inps, cached_oups, config):
    global USE_LINK
    global USE_DDP
    device = next(subgraph.parameters()).device
    w_para = []
    b_para = []
    optimizer = None
    for name, layer in subgraph.named_modules():
        if isinstance(layer, _SUPPORT_MODULE_TYPES):
            w_para += [layer.weight]
            if layer.bias is not None:
                b_para += [layer.bias]
            
    if 'bias_lr' in config and len(b_para) != 0:
        optimizer = torch.optim.Adam([
                {'params': w_para, 'lr': config.weight_lr},
                {'params': b_para, 'lr': config.bias_lr}
            ])
    else:
        optimizer = torch.optim.Adam(w_para, lr=config.weight_lr)

    loss_func = LossFunction(subgraph)

    if any([USE_DDP, USE_LINK]):
        world_size = link.get_world_size() if USE_LINK else dist.get_world_size()
    else:
        world_size = 1

    logger.info('The world size is {}.'.format(world_size))
    '''start training'''
    logger.info('start tuning by adasparse')
    sz = len(cached_inps)
    for i in range(config.max_count):
        idx = np.random.randint(0, sz)
        cur_inp = cached_inps[idx]
        cur_inp = [inp.to(device) for inp in cur_inp]
        cur_out = cached_oups[idx].to(device)
        optimizer.zero_grad()
        out_sparse = subgraph(*cur_inp)
        err = loss_func(out_sparse, cur_out)
        err /= world_size
        if world_size > 1:
            if world_size > 1:
                if USE_LINK:
                    link.allreduce(err)
                elif USE_DDP:
                    dist.all_reduce(err)

        err.backward()
        optimizer.step()
        if err < 0.00001:
            logger.info("current rec loss is small, so early stop!")
            break
    torch.cuda.empty_cache()


def extract_subgraph(orig_module: nn.Module, nodes: List[fx.Node], inputs: List[fx.Node], extra_inputs: List[fx.Node], output: fx.Node):
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env = dict()
    for input in set(inputs + extra_inputs):
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node
    for node in nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    # create this or there will not be return value
    new_graph.output(env[output])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)


def find_num_nodes(nodes):
    num = 0
    for node in nodes:
        if isinstance(node, Node):
            num += 1
    return num


# Recommend: log this to check if the layer is right. You can define your own layer manually or automatically like this
# extract the linked-list/single-chain
def extract_layer(node, dense_modules):
    layer_node_list = []
    cur_node = node
    is_next_block = False  # check whether stoped by a block
    while True:
        logger.debug('cur_node in layer is {}'.format(cur_node))
        layer_node_list.append(cur_node)  # valid node here
        stop = (len(cur_node.users) == 0)
        for user in cur_node.users:
            if user.op == 'call_module' and isinstance(dense_modules[user.target], _SUPPORT_MODULE_TYPES):
                stop = True
            # TODO: only short-cut here, consider more here
            # TODO: can also use un/completed to check here.
            if ('add' in user.name and user.op in ['call_function', 'call_method']):
                stop = True
            if ('cat' in user.name and user.op in ['call_function', 'call_method']):
                stop = True
            if user.op == 'output':
                is_next_block, stop = True, True
        if stop:
            break
        cur_node = list(cur_node.users.keys())[0]
    if find_num_nodes(cur_node.users) > 1:
        is_next_block = True
    return layer_node_list, is_next_block


# Recommend: log this to check if the block is right. You can define your own block manually or automatically like this
# extract the block one such as short-cut
def extract_block(input_nodes, dense_modules, depth=0):
    if depth > 2:
        # stack 2 or 3 layers for no short-cut structure
        return []
    layer_node_list = []
    is_block = False
    cnt = dict()
    q, p = [], []  # q records the completed node, p records the uncompleted nodes
    for input in input_nodes:
        for user in input.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
    while len(q) != 0:
        cur_node = q.pop(0)  # valid node here
        logger.debug('cur node is {}'.format(cur_node))
        if len(p) == 0 and len(q) == 0:
            break
        layer_node_list.append(cur_node)
        for user in cur_node.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
        logger.debug('uncompleted nodes are {}'.format(p))
    exp_nodes, is_next_block = extract_layer(cur_node, dense_modules)
    if is_block or is_next_block:
        return layer_node_list + exp_nodes
    else:
        return layer_node_list + exp_nodes + extract_block([exp_nodes[-1]], dense_modules, depth + 1)


def do_bias_correction_v2(dense_inp_module, weight, dense_inps, dense_oups):
    device = next(dense_inp_module.parameters()).device

    """ test do_bias_correction_v2 is right

    a = [torch.rand(4,5,6)]
    b = [torch.rand(4,5,6)]
    c = [torch.rand(4,5,6)]
    dense_inps = [a,b,c]


    inp = []
    sum_tem = 0
    for li in dense_inps:
        tmp = torch.stack(li, dim=0)
        tmp = torch.mean(tmp, dim=0, keepdim=False)
        sum_tem += tmp
    inp = sum_tem / len(dense_inps)
    inp = torch.mean(inp, dim=0, keepdim=True)
    print(inp)

    inp = []
    for li in dense_inps:
        inp.append(torch.stack(li, dim=0))
    inp = torch.stack(inp)
    inp = torch.mean(inp, dim=[0, 1], keepdim=False)
    inp = torch.mean(inp, dim=0, keepdim=True)
    print(inp)
    """

    inp = []
    sum_tmp = 0
    for li in dense_inps:
        tmp = torch.stack(li, dim=0)
        tmp = torch.mean(tmp, dim=0, keepdim=False)
        sum_tmp += tmp
    inp = sum_tmp / len(dense_inps)
    inp = torch.mean(inp, dim=0, keepdim=True)
    bias_shift = None
    if isinstance(dense_inp_module, torch.nn.Linear):
        no_bias_op_out = torch.nn.functional.linear(inp.to(device),
                                               weight=weight,
                                               bias=None
                                            )
        sum_tmp = 0
        for li in dense_oups:
            tmp = torch.mean(li, dim=0)
            sum_tmp += tmp
        out = sum_tmp / len(dense_oups)
        no_bias_op_out = torch.mean(no_bias_op_out, dim=1)
        bias_shift = out.to(device) - no_bias_op_out
    else:
        no_bias_op_out = torch.nn.functional.conv2d(input=inp.to(device),
                                               weight=weight,
                                               bias=None,
                                               stride=dense_inp_module.stride,
                                               padding=dense_inp_module.padding,
                                               dilation=dense_inp_module.dilation,
                                               groups=dense_inp_module.groups
                                            )
        sum_tmp = 0
        for li in dense_oups:
            tmp = torch.mean(li, dim=[0, 2, 3])
            sum_tmp += tmp
        out = sum_tmp / len(dense_oups)
        no_bias_op_out = torch.mean(no_bias_op_out, dim=[0, 2, 3])
        bias_shift = out.to(device) - no_bias_op_out
    return bias_shift


def do_bias_correction(dense_inp_module, weight, dense_inps, dense_oups):
    device = next(dense_inp_module.parameters()).device

    inp = []
    for li in dense_inps:
        inp.append(torch.stack(li, dim=0))
    inp = torch.stack(inp)
    inp = torch.mean(inp, dim=[0, 1], keepdim=False)
    inp = torch.mean(inp, dim=0, keepdim=True)
    bias_shift = None
    if isinstance(dense_inp_module, torch.nn.Linear):
        no_bias_op_out = torch.nn.functional.linear(inp.to(device),
                                               weight=weight,
                                               bias=None
                                            )
        out = torch.stack(dense_oups)
        out = torch.mean(out, dim=[0, 1])
        no_bias_op_out = torch.mean(no_bias_op_out, dim=1)
        bias_shift = out.to(device) - no_bias_op_out
    else:
        no_bias_op_out = torch.nn.functional.conv2d(input=inp.to(device),
                                               weight=weight,
                                               bias=None,
                                               stride=dense_inp_module.stride,
                                               padding=dense_inp_module.padding,
                                               dilation=dense_inp_module.dilation,
                                               groups=dense_inp_module.groups
                                            )
        out = torch.stack(dense_oups)
        out = torch.mean(out, dim=[0, 1, 3, 4])
        no_bias_op_out = torch.mean(no_bias_op_out, dim=[0, 2, 3])
        bias_shift = out.to(device) - no_bias_op_out
    return bias_shift


def pts_reconstruction(model: GraphModule, cali_data: list, config: dict):
    r"""
    Basic optimization objective:

    Args:
        model (torch.nn.Module): a prepared GraphModule to do PTQ
        cali_data (List): a list of calibration tensor
        config (dict): a config for PTS reconstruction

    >>> sample config : {
            pattern: block (str, Available options are [layer, block].)
            scale_lr: 4.0e-5 (learning rate for learning step size of activation)
            warm_up: 0.2 (0.2 * max_count iters without regularization to floor or ceil)
            weight: 0.01 (loss weight for regularization item)
            max_count: 20000 (optimization iteration)
            b_range: [20,2] (beta decaying range )
            keep_gpu: True (calibration data restore in gpu or cpu)
            round_mode: learned_hard_sigmoid (ways to reconstruct the weight, currently only support learned_hard_sigmoid)
            prob: 0.5 (dropping probability of QDROP)
        }

    """
    # assert model is on cuda
    if not config.keep_gpu:
        cali_data = [inp.cpu() for inp in cali_data]
    '''set state first'''

    dense_model = model
    dense_model.eval()
    sparse_model = deepcopy_graphmodule(model)
    before_run(sparse_model)
    sparse_model.eval()
    disable_sparsification(dense_model)
    enable_sparsification(sparse_model)
    torch.cuda.empty_cache()
    nodes = list(sparse_model.graph.nodes)
    dense_modules = dict(dense_model.named_modules())
    sparse_modules = dict(sparse_model.named_modules())
    checked_nodes = dict()
    if hasattr(config, 'bias_correction') and config.bias_correction == True:
        logger.info("do bias correction!")
        for node in nodes:
            if node.op == "call_module" and isinstance(dense_modules[node.target], _SUPPORT_MODULE_TYPES):
                layer_node_list = [node]
                extra_inputs = append_extra_inputs(nodes, layer_node_list)
                cur_node = find_cur_node(layer_node_list)
                dense_module = dense_modules[cur_node.target]
                dense_inp_module = dense_modules[node.target]
                sparse_module = sparse_modules[node.target]

                dense_inps, dense_oups = save_inp_oup_data(dense_model, dense_inp_module, dense_module, cali_data,
                                                            store_inp=True, store_oup=True, keep_gpu=config.keep_gpu)
                weight = sparse_module.weight_fake_sparse.generate_mask(sparse_module.weight) * sparse_module.weight
                bias_shift = do_bias_correction_v2(dense_inp_module, weight, dense_inps, dense_oups)
                if sparse_module.bias is not None:
                    sparse_module.bias.data = bias_shift

    for node in nodes:
        if node in checked_nodes:
            continue
        if node.op == "call_module" and isinstance(dense_modules[node.target], _SUPPORT_MODULE_TYPES):
            logger.info('prepare {} reconstruction for {}'.format(config.pattern, node.target))
            if config.pattern == 'independent':
                layer_node_list = [node]
            elif config.pattern == 'layer':
                layer_node_list, _ = extract_layer(node, dense_modules)
            elif config.pattern == 'block':
                layer_node_list = extract_block(node.all_input_nodes, dense_modules)
            else:
                raise NotImplementedError
            extra_inputs = append_extra_inputs(nodes, layer_node_list)
            cur_node = find_cur_node(layer_node_list)
            dense_module = dense_modules[cur_node.target]
            dense_inp_module = dense_modules[node.target]
            sparse_module = sparse_modules[node.target]

            
            dense_inps, dense_oups = save_inp_oup_data(dense_model, dense_inp_module, dense_module, cali_data,
                                                     store_inp=config.use_dense_inps, store_oup=True, keep_gpu=config.keep_gpu)
            sparse_inps, _ = save_inp_oup_data(sparse_model, sparse_module, None, cali_data, store_inp=(not config.use_dense_inps),
                                               store_oup=False, keep_gpu=config.keep_gpu)
            logger.info('the node list is below!')
            logger.info(layer_node_list)
            subgraph = extract_subgraph(sparse_modules, layer_node_list, node.all_input_nodes, extra_inputs, cur_node)
            logger.info(subgraph)
            cached_inps = None
            if config.use_dense_inps:
                cached_inps = dense_inps
                logger.info('use dense_inps!!!')
            else:
                cached_inps = sparse_inps
                logger.info('use sparse_inps!!!')
            cached_oups = dense_oups
            subgraph_reconstruction(subgraph, cached_inps, cached_oups, config)
            for x in layer_node_list:
                checked_nodes[x] = True
    return sparse_model


if __name__ == '__main__':
    from torchvision.models import resnet
    model = resnet.resnet18(pretrained=True)
