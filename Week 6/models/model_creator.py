""" Functions to create models """

import torch
import torch.nn as nn
import torchvision
from fvcore.nn import FlopCountAnalysis

def create(model_name: str, load_pretrain: bool, num_classes: int) -> nn.Module:
    if model_name == 'x3d_xs':
        return create_x3d_xs(load_pretrain, num_classes)
    
    elif model_name == 'x3d_m':
        return create_x3d_m(load_pretrain, num_classes)
    
    elif model_name == 'x3d_l':
        return create_x3d_l(load_pretrain, num_classes)
    
    elif model_name == 'x3d_xxs':
        return create_x3d_xxs(load_pretrain, num_classes)
    
    elif model_name == 'mvit_v1_b':
        return create_mvit_v1_b(load_pretrain, num_classes)

    elif model_name == 'mvit_v2_s':
        return create_mvit_v2_s(load_pretrain, num_classes)
    
    elif model_name == 'mvit_v2_xs':
        return create_mvit_v2_xs(load_pretrain, num_classes)
    
    elif model_name == 'slow_r50':
        return create_slow_r50(load_pretrain, num_classes)
    
    # elif model_name == 'slowfast_r50':
        # return create_slowfast_r50(load_pretrain, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported")
    
# def create_slowfast_r50(load_pretrain, num_classes):
#     # best parameters for slowfast_r50 are 8x8 (clip-length x stride) (they call it frame length x sample rate)
#     # and resolution 256x256 and num_segments = 10 and num crops = 3
#     model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=load_pretrain)
#     model.blocks[6].proj = nn.Linear(2304, num_classes, bias=True)
#     return model


def create_slow_r50(load_pretrain, num_classes):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Linear(2048, num_classes, bias=True)
    return model
    
def create_x3d_xxs(load_pretrain, num_classes):
    # best parameters for x3d_xs are 4x12 (clip-length x stride) (they call it frame length x sample rate)
    # and resolution 160x160
    from models.x3d import create_x3d as create_x3d_xxs
    return create_x3d_xxs()

def create_x3d_xs(load_pretrain, num_classes):
    # best parameters for x3d_xs are 4x12 (clip-length x stride) (they call it frame length x sample rate)
    # and resolution 182x182 (en vd son 160x160 pero el profe se equivocó y utilizó 182x182)
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def create_x3d_m(load_pretrain, num_classes):
    # best parameters for x3d_m are 16x5 (clip-length x stride) (they call it frame length x sample rate)
    # and resolution 224x224
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def create_x3d_l(load_pretrain, num_classes):
    # best parameters for x3d_l are 16x5 (clip-length x stride) (they call it frame length x sample rate)
    # and resolution 312x312
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=load_pretrain)
    model.blocks[5].proj = nn.Identity()
    return nn.Sequential(
        model,
        nn.Linear(2048, num_classes, bias=True),
    )

def create_mvit_v1_b(load_pretrain, num_classes):
    # best parameters for mvit_v2_s are 16x7.5 (clip-length x stride) (they call it frame length x sample rate)
    # and num_segments = 5 and resolution 224x224
    model = torchvision.models.video.mvit_v1_b(weights='KINETICS400_V1' if load_pretrain else None)
    model.head[1] = torch.nn.Linear(768, num_classes, bias=True)

    return model
    
def create_mvit_v2_s(load_pretrain, num_classes):
    # best parameters for mvit_v2_s are 16x7.5 (clip-length x stride) (they call it frame length x sample rate)
    # and num_segments = 5 and resolution 224x224
    model = torchvision.models.video.mvit_v2_s(weights='KINETICS400_V1' if load_pretrain else None)
    model.head[1] = torch.nn.Linear(768, num_classes, bias=True)

    return model

def create_mvit_v2_xs(load_pretrain, num_classes):
    # best parameters for mvit_v2_s are 4x12 (clip-length x stride) (they call it frame length x sample rate)
    # and num_segments = 5 and resolution 160x160
    from models.mvit import mvit_v2_s as mvit_v2_xs
    model = mvit_v2_xs(weights=None)
    model.head[1] = torch.nn.Linear(384, num_classes, bias=True)

    return model
    