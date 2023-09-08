""" ConvNeXt

Paper: `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

Original code and weights from https://github.com/facebookresearch/ConvNeXt, original copyright below

Model defs atto, femto, pico, nano and _ols / _hnf variants are timm specific.

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import named_apply, build_model_with_cfg, checkpoint_seq

from timm.models.layers.helpers import to_2tuple #MLP调用

from timm.models.layers import trunc_normal_, SelectAdaptivePool2d, DropPath, LayerNorm2d, LayerNorm, \
    create_conv2d, get_act_layer, make_divisible, to_ntuple
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

from timm.models.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from timm.models.vision_transformer import checkpoint_filter_fn, get_init_weights_vit

from model.swintransformer_new2 import window_partition, window_reverse, get_relative_position_index, WindowAttention,SwinTransformerBlock, PatchMerging, BasicLayer

from model.cbam import *
from model.IAFF import *
from model.ECA import *
from model.AFF import *
# import model.SE

__all__ = ['ConvNeXt']  # model_registry will add each entrypoint fn to this


""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()

    channels_per_group = num_channels // groups

    # reshape input tensor
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # transpose dimensions
    x = torch.transpose(x, 1, 2).contiguous()

    # reshape tensor back to original size
    x = x.view(batch_size, -1, height, width)

    return x

class C_Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }

default_cfgs = dict(
    # timm specific variants
    convnext_atto=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_atto_ols=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_ols_a2-78d1c8f3.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_femto=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_femto_ols=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_ols_d1-246bf2ed.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_pico=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth',
        test_input_size=(3, 288, 288), test_crop_pct=0.95),
    convnext_pico_ols=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_ols_d1-611f0ca7.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_nano=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_nano_ols=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_ols_d1h-ae424a9a.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_tiny_hnf=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_tiny_hnf_a2h-ab7e9df2.pth',
        crop_pct=0.95, test_input_size=(3, 288, 288), test_crop_pct=1.0),

    convnext_tiny=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_small=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_base=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_large=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    convnext_tiny_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_small_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_base_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_large_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),
    convnext_xlarge_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth',
        test_input_size=(3, 288, 288), test_crop_pct=1.0),

    convnext_tiny_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_small_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_base_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_large_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_xlarge_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    convnext_tiny_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth", num_classes=21841),
    convnext_small_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth", num_classes=21841),
    convnext_base_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth", num_classes=21841),
    convnext_large_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth", num_classes=21841),
    convnext_xlarge_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth", num_classes=21841),
)


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        in_chs (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chs,
            out_chs=None,
            kernel_size=7,
            stride=1,
            dilation=1,
            mlp_ratio=4,
            conv_mlp=False,
            conv_bias=True,
            ls_init_value=1e-6,
            act_layer='gelu',
            norm_layer=None,
            drop_path=0.,
    ):
        super().__init__()
        out_chs = out_chs or in_chs
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = ConvMlp if conv_mlp else C_Mlp
        self.use_conv_mlp = conv_mlp

        self.conv_dw = create_conv2d(
            in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, depthwise=True, bias=conv_bias)

        self.norm = norm_layer(out_chs)

        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.eca = ECABlock(out_chs)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        x = self.eca(x)

        x = self.drop_path(x)
        x = x + shortcut
        return x

class ConvNeXtStage(nn.Module):

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=7,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            act_layer='gelu',
            norm_layer=None,
            norm_layer_cl=None
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs, out_chs, kernel_size=ds_ks, stride=stride,
                    dilation=dilation[0], padding=pad, bias=conv_bias),
            )
            # self.cbam = CBAM(out_chs)
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()
            # self.cbam = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(ConvNeXtBlock(
                in_chs=in_chs,
                out_chs=out_chs,
                kernel_size=kernel_size,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                act_layer=act_layer,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        # x = self.cbam(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

class downsample1(nn.Module):
    def __init__(
            self, dim, out_dim, input_resolution, norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            output_stride=32,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            kernel_sizes=7,
            ls_init_value=1e-6,
            stem_type='patch',
            patch_size=4,
            head_init_scale=1.,
            head_norm_first=False,
            conv_mlp=False,
            conv_bias=True,
            act_layer='gelu',
            norm_layer=None,
            drop_rate=0.,
            drop_path_rate=0.1,
            # swin
            img_size=224, embed_dim=96, depths_swin=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            head_dim=None, window_size=7, mlp_ratio=4., qkv_bias=True, attn_drop_rate=0.,
            ape=False, patch_norm=True, weight_init='', norm_layer_swin=nn.LayerNorm,
    ):
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            norm_layer_cl = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        # swin
        self.num_layers_swin = len(depths_swin)
        self.embed_dim = embed_dim
        self.num_features_swin = int(embed_dim * 2 ** (self.num_layers_swin - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer_swin if patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers_swin)]
        embed_out_dim = embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers_swin)(head_dim)
        window_size = to_ntuple(self.num_layers_swin)(window_size)
        mlp_ratio = to_ntuple(self.num_layers_swin)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_swin))]  # stochastic depth decay rule
        layers = []

        for i in range(self.num_layers_swin):
            layers += [BasicLayer(
                dim=embed_dim[i],
                out_dim=embed_out_dim[i],
                input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                depth=depths_swin[i],
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths_swin[:i]):sum(depths_swin[:i + 1])],
                norm_layer=norm_layer_swin,
                downsample=PatchMerging if (i < self.num_layers_swin - 1) else None
            )]
        self.layers = nn.Sequential(*layers)

        downsample2 = []
        for i in range(self.num_layers_swin):
            downsample2 += [downsample1(
                dim=embed_dim[i],
                out_dim=embed_out_dim[i],
                input_resolution=(self.patch_grid[0] // (2 ** i), self.patch_grid[1] // (2 ** i)),
                norm_layer=norm_layer_swin,
                downsample=PatchMerging if (i < self.num_layers_swin - 1) else None
            )]
        self.downsample2 = nn.Sequential(*downsample2)

        if weight_init != 'skip':
            self.init_weights_swin(weight_init)

        # convnext
        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(dims[0])
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs,
                out_chs,
                kernel_size=kernel_sizes[i],
                stride=stride,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl
            ))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)

        # conv = []
        # for i in range(4):
        #     conv += [create_conv2d(dims[i]*2, dims[i], kernel_size=1, bias=True)]
        # self.conv = nn.Sequential(*conv)

        iaff = []
        for i in range(4):
            iaff += [AFF(dims[i])]
        self.iaff = nn.Sequential(*iaff)

        # self.afp = AFP()

        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def init_weights_swin(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        if self.absolute_pos_embed is not None:
            trunc_normal_(self.absolute_pos_embed, std=.02)
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )
    @torch.jit.ignore
    def group_matcher_swin(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def set_grad_checkpointing_swin(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x1 = self.stem(x)

        x2 = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x2 = x2 + self.absolute_pos_embed
        x2 = self.pos_drop(x2)

        for i in range(4):
            x1 = self.stages[i](x1)
            B, C, H, W = x1.shape
            x2 = self.layers[i](x2)
            x3 = x2.view(B, H, W, C)

            x3 = x3.permute(0, 3, 1, 2).contiguous()
            # x1 = torch.cat((x1, x3), dim=1)
            # x1 = channel_shuffle(x1, groups=4)
            # x1 = self.conv[i](x1)

            x1 = self.iaff[i](x1, x3)

            # x1 = x1 + x2
            x2 = self.downsample2[i](x2)

        x = self.norm_pre(x1)
        return x
#让我提交个东西呢
    def forward_head(self, x, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            # print(name)
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        # print(name)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)

def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v
    return out_dict

def _create_convnext(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        ConvNeXt, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


@register_model
def convnext_atto(pretrained=False, **kwargs):
    # timm femto variant (NOTE: still tweaking depths, will vary between 3-4M param, current is 3.7M
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_atto', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_atto_ols(pretrained=False, **kwargs):
    # timm femto variant with overlapping 3x3 conv stem, wider than non-ols femto above, current param count 3.7M
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), conv_mlp=True, stem_type='overlap_tiered', **kwargs)
    model = _create_convnext('convnext_atto_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_femto(pretrained=False, **kwargs):
    # timm femto variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_femto', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_femto_ols(pretrained=False, **kwargs):
    # timm femto variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), conv_mlp=True, stem_type='overlap_tiered', **kwargs)
    model = _create_convnext('convnext_femto_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_pico(pretrained=False, **kwargs):
    # timm pico variant
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_pico', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_pico_ols(pretrained=False, **kwargs):
    # timm nano variant with overlapping 3x3 conv stem
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), conv_mlp=True,  stem_type='overlap_tiered', **kwargs)
    model = _create_convnext('convnext_pico_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_nano(pretrained=False, **kwargs):
    # timm nano variant with standard stem and head
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_nano', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_nano_ols(pretrained=False, **kwargs):
    # experimental nano variant with overlapping conv stem
    model_args = dict(
        depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), conv_mlp=True, stem_type='overlap', **kwargs)
    model = _create_convnext('convnext_nano_ols', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny_hnf(pretrained=False, **kwargs):
    # experimental tiny variant with norm before pooling in head (head norm first)
    model_args = dict(
        depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_small(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                      patch_size=4, window_size=7, embed_dim=128, depths_swin=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                      **kwargs)
    model = _create_convnext('convnext_base', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_small_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xlarge_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_xlarge_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_small_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_large_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xlarge_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_xlarge_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_tiny_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_small_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base_in22k(pretrained=False, **kwargs):

    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                      patch_size=4, window_size=7, embed_dim=128, depths_swin=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                      **kwargs)
    model = _create_convnext('convnext_base_in22k', pretrained=pretrained, **model_args)

    return model


@register_model
def convnext_large_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_large_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_xlarge_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_xlarge_in22k', pretrained=pretrained, **model_args)
    return model
