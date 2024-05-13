import torch
import torch.nn as nn


def get_select_filter_index(weight):
    assert len(weight.shape) == 4
    non_zero_channels_mask = weight.sum(dim=(1, 2, 3)) != 0
    select_index = torch.where(non_zero_channels_mask)[0].tolist()
    select_index.sort()
    return select_index


def export_resnet_model(model, oristate_dict, layer, args):
    if len(args.gpu) > 1:
        name_base = "module."
    else:
        name_base = ""

    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    cnt = 1
    for layer, num in enumerate(current_cfg):
        layer_name = "layer" + str(layer + 1) + "."
        for k in range(num):
            for l in range(2):
                cnt += 1
                cov_id = cnt

                conv_name = layer_name + str(k) + ".conv" + str(l + 1)
                conv_weight_name = conv_name + ".weight"
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base + conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    select_index = get_select_filter_index(
                        oristate_dict[conv_weight_name]
                    )

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + conv_weight_name][index_i][
                                    index_j
                                ] = oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base + conv_weight_name][
                                index_i
                            ] = oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + conv_weight_name][index_i][
                                index_j
                            ] = oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base + conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace("module.", "")

        if isinstance(module, nn.Conv2d):
            conv_name = name + ".weight"
            if "shortcut" in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base + conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base + name + ".weight"] = oristate_dict[name + ".weight"]
            state_dict[name_base + name + ".bias"] = oristate_dict[name + ".bias"]

    model.load_state_dict(state_dict)
