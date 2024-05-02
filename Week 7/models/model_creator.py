""" Functions to create models """

import torch
import torch.nn as nn
import torchvision
from fvcore.nn import FlopCountAnalysis

def create(model_name: str, load_pretrain: bool, num_classes: int, input_channels: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes, input_channels)
    
    elif model_name == 'x3d_m':
        return create_x3d_m(load_pretrain, num_classes, input_channels)

    else:
        raise ValueError(f"Model {model_name} not supported")

    
def create_x3d_m(load_pretrain, num_classes, input_channels):
    # best parameters for x3d_m are 16x5 (clip-length x stride) (they call it frame length x sample rate)
    # and resolution 224x224
    from models.x3d import create_x3d
    return create_x3d(input_channel=input_channels)

def create_x3d_xs(load_pretrain, num_classes, input_channels):
    # best parameters for x3d_xs are 4x12 (clip-length x stride) (they call it frame length x sample rate)
    # and resolution 182x182 (en vd son 160x160 pero el profe se equivocó y utilizó 182x182)
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )