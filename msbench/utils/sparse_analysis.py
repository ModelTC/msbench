import torch.nn as nn


def analysis_model_sparsity(model):
    sparsity_dict = {}
    total_num = 0
    total_non_zero_num = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.conv._ConvNd)\
           or isinstance(module, nn.Linear):
            module_type = type(module)
            element_num = module.weight.numel()
            non_zero_num = module.weight.count_nonzero()
            if module.bias is not None:
                element_num += module.bias.numel()
                non_zero_num += module.bias.count_nonzero()
            sparsity_dict[name] = {'element_num': element_num,
                                   'non_zero_num': non_zero_num,
                                   'module_type': module_type}
            total_non_zero_num += non_zero_num
            total_num += element_num
    sparsity_dict['model'] = {'total_num': total_num,
                              'total_non_zero_num': total_non_zero_num}
    return sparsity_dict


if __name__ == '__main__':
    import timm
    model_list = timm.list_models(pretrained=True)
    for model_name in model_list:
        print(model_name)

    model = timm.create_model('xcit_small_12_p8_224', pretrained=True)
    sparsity_dict = analysis_model_sparsity(model)
    for name, value in sparsity_dict.items():
        if name != "model":
            print("layer_name is {}, module type is {}, sparsity = {}".format(
                  name,
                  value["module_type"],
                  1 - value["non_zero_num"] / value["element_num"]))
        else:
            print("Whole Network, sparsity = {}".format(1 - value["total_non_zero_num"] / value["total_num"]))  # noqa W504
