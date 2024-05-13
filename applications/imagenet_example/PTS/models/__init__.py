import torch
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .resnet import (  # noqa: F401
    resnet18, resnet34, resnet50, resnet101, resnet152
)
from .mobilenet_v2 import mobilenet_v2
from spring.models import SPRING_MODELS_REGISTRY


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(config):
    if config['type'] in SPRING_MODELS_REGISTRY.query():
        model = SPRING_MODELS_REGISTRY.get(config['type'])(task='classification', **config['kwargs'])
    else:
        model = globals()[config['type']](**config['kwargs'])
    checkpoint = torch.load(config.path, map_location='cpu')
    checkpoint = remove_prefix(checkpoint, 'module.')
    model.load_state_dict(checkpoint)
    return model
