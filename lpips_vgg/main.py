"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from collections import namedtuple
import os
from torchvision.models import VGG16_Weights
from torchvision import models
from pathlib import Path

os.environ['TORCH_HOME'] = '/gpfs3/well/papiez/users/zwk579/.cache/torch/hub/checkpoints/'

class LossTerm(nn.Module):
    def __init__(self,weight=1.0,factor=1.0,start_iter=None,start_epoch=None):
        super(LossTerm, self).__init__()
        self.weight = weight
        self.factor = factor
        self.start_iter = start_iter
        self.start_epoch = start_epoch


class LPIPS(LossTerm):
    def __init__(
        self,
        perceptual_weight=1.0,
        perceptual_factor=1.0,
        perceptual_start_iter=None,
        perceptual_start_epoch=None,
        use_dropout=True,
        eval_mode=True,
        pretrained=True,
        requires_grad=False,
        **kwargs
    ):
        super().__init__(
            perceptual_weight,
            perceptual_factor,
            perceptual_start_iter,
            perceptual_start_epoch
        )

        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.load_from_pretrained()

        self.scaling_layer = ScalingLayer() 
        self.net = vgg16(pretrained=True, requires_grad=False)

        # Freeze everything by default
        for p in self.parameters():
            p.requires_grad = False

        if eval_mode:
            self.eval()
        
    def load_from_pretrained(self, name="vgg_lpips", ignore_keys=['']):
        ckpt = Path(__file__).parent / "lpips_weights" / "vgg.pth"
        ckpt = ckpt.resolve()
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True),
            strict=True
        )
        print(f"loaded pretrained LPIPS loss from {ckpt}")


    def forward(self, input, target):
        in0, in1 = self.scaling_layer(input), self.scaling_layer(target)

        outs0 = self.net(in0)
        outs1 = self.net(in1)

        res = []

        for kk in range(len(self.chns)):
            feat0 = normalize_tensor(outs0[kk])
            feat1 = normalize_tensor(outs1[kk])

            diff = (feat0 - feat1) ** 2
            scaled = self.lins[kk].model(diff)
            scaled_avg = spatial_average(scaled, keepdim=True)

            res.append(scaled_avg)

        val = sum(res)
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

