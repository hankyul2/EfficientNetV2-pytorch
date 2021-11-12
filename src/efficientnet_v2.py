import copy
from functools import partial

import torch
from torch import nn

from src.pretrained_weight_loader import load_from_zoo


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups, norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False), norm_layer(out_channel), act())


class SEUnit(nn.Module):
    def __init__(self, in_channel, reduction_ratio=4, act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(self.survival).to(x.device)


class MBConvConfig:
    def __init__(self, expand_ratio, kernel, stride, in_ch, out_ch, layers, use_se, fused, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.act = act
        self.norm_layer = norm_layer
        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        return divisible_channel


class MBConv(nn.Module):
    def __init__(self, config, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = config.adjust_channels(config.in_ch, config.expand_ratio)
        block = []

        if config.fused:
            block.append(ConvBNAct(config.in_ch, inter_channel, config.kernel, config.stride, 1, config.norm_layer, config.act))
        else:
            block.append(ConvBNAct(config.in_ch, inter_channel, 1, 1, 1, config.norm_layer, config.act))
            block.append(ConvBNAct(inter_channel, inter_channel, config.kernel, config.stride, inter_channel, config.norm_layer, config.act))

        if config.use_se:
            block.append(SEUnit(inter_channel))

        block.append(ConvBNAct(inter_channel, config.out_ch, 1, 1, 1, config.norm_layer, nn.Identity))
        self.block = nn.Sequential(*block)
        self.use_skip_connection = config.stride == 1 and config.in_ch == config.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class EfficientNetV2(nn.Module):
    def __init__(self, layer_infos, dropout=0.2, stochastic_depth=0.0, block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.last_channels = layer_infos[-1].out_ch
        self.out_channels = self.last_channels * 4

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.features = nn.Sequential(
            ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act),
            *self.make_stages(layer_infos, block),
            ConvBNAct(self.last_channels, self.out_channels, 1, 1, 1, self.norm_layer, self.act)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

    def make_stages(self, layer_infos, block):
        return [nn.Sequential(*self.make_layers(copy.copy(layer_info), block)) for layer_info in layer_infos]

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x):
        return self.dropout(torch.flatten(self.avg_pool(self.features(x)), 1))


def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


def get_efficientnet_v2(model_name, pretrained, **kwargs):
    # Todo: Change residual config
    # reference(pytorch): https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    # reference(official): https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py

    if 'efficientnet_v2_s' in model_name:
        dropout = 0.2
    elif 'efficientnet_v2_m' in model_name:
        dropout = 0.2
    elif 'efficientnet_v2_l' in model_name:
        dropout = 0.3
    elif 'efficientnet_v2_xl' in model_name:
        dropout = 0.3

    residual_config = [
        #    expand k  s  in  out layers  se  fused
        MBConvConfig(1, 3, 1, 24, 24, 2, use_se=False, fused=True),
        MBConvConfig(4, 3, 2, 24, 48, 4, use_se=False, fused=True),
        MBConvConfig(4, 3, 2, 48, 64, 4, use_se=False, fused=True),
        MBConvConfig(4, 3, 2, 64, 128, 6, use_se=True, fused=False),
        MBConvConfig(6, 3, 1, 160, 160, 9, use_se=True, fused=False),
        MBConvConfig(6, 5, 2, 160, 256, 15, use_se=True, fused=False),
    ]

    model = EfficientNetV2(residual_config, dropout=dropout, stochastic_depth=0.2, block=MBConv, act_layer=nn.SiLU)
    efficientnet_v2_init(model)

    if pretrained:
        load_from_zoo(model, model_name)

    return model