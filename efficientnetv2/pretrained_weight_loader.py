import os
import re
import subprocess
from pathlib import Path

import numpy as np

import torch


model_urls = {
    "efficientnet_v2_s": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-s.npy",
    "efficientnet_v2_m": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-m.npy",
    "efficientnet_v2_l": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-l.npy",
    "efficientnet_v2_s_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-s-21k.npy",
    "efficientnet_v2_m_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-m-21k.npy",
    "efficientnet_v2_l_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-l-21k.npy",
    "efficientnet_v2_xl_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-xl-21k.npy",
}


def load_from_zoo(model, model_name, pretrained_path='pretrained/official'):
    Path(os.path.join(pretrained_path, model_name)).mkdir(parents=True, exist_ok=True)
    file_name = os.path.join(pretrained_path, model_name, os.path.basename(model_urls[model_name]))
    load_npy(model, load_npy_from_url(url=model_urls[model_name], file_name=file_name))


def load_npy_from_url(url, file_name):
    if not Path(file_name).exists():
        subprocess.run(["wget", "-r", "-nc", '-O', file_name, url])
    return np.load(file_name, allow_pickle=True).item()


def npz_dim_convertor(name, weight):
    weight = torch.from_numpy(weight)
    if 'kernel' in name:
        if weight.dim() == 4:
            if weight.shape[3] == 1:
                # depth-wise convolution 'h w in_c out_c -> in_c out_c h w'
                weight = torch.permute(weight, (2, 3, 0, 1))
            else:
                # 'h w in_c out_c -> out_c in_c h w'
                weight = torch.permute(weight, (3, 2, 0, 1))
        elif weight.dim() == 2:
            weight = weight.transpose(1, 0)
    elif 'scale' in name or 'bias' in name:
        weight = weight.squeeze()
    return weight


def load_npy(model, weight):
    name_convertor = [
        # stem
        ('stem.0.weight', 'stem/conv2d/kernel/ExponentialMovingAverage'),
        ('stem.1.weight', 'stem/tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('stem.1.bias', 'stem/tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('stem.1.running_mean', 'stem/tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('stem.1.running_var', 'stem/tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # fused layer
        ('block.fused.0.weight', 'conv2d/kernel/ExponentialMovingAverage'),
        ('block.fused.1.weight', 'tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('block.fused.1.bias', 'tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('block.fused.1.running_mean', 'tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('block.fused.1.running_var', 'tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # linear bottleneck
        ('block.linear_bottleneck.0.weight', 'conv2d/kernel/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.weight', 'tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.bias', 'tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.running_mean', 'tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.running_var', 'tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # depth wise layer
        ('block.depth_wise.0.weight', 'depthwise_conv2d/depthwise_kernel/ExponentialMovingAverage'),
        ('block.depth_wise.1.weight', 'tpu_batch_normalization_1/gamma/ExponentialMovingAverage'),
        ('block.depth_wise.1.bias', 'tpu_batch_normalization_1/beta/ExponentialMovingAverage'),
        ('block.depth_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage'),
        ('block.depth_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage'),

        # se layer
        ('block.se.fc1.weight', 'se/conv2d/kernel/ExponentialMovingAverage'), ('block.se.fc1.bias', 'se/conv2d/bias/ExponentialMovingAverage'),
        ('block.se.fc2.weight', 'se/conv2d_1/kernel/ExponentialMovingAverage'), ('block.se.fc2.bias', 'se/conv2d_1/bias/ExponentialMovingAverage'),

        # point wise layer
        ('block.fused_point_wise.0.weight', 'conv2d_1/kernel/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.weight', 'tpu_batch_normalization_1/gamma/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.bias', 'tpu_batch_normalization_1/beta/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage'),

        ('block.point_wise.0.weight', 'conv2d_1/kernel/ExponentialMovingAverage'),
        ('block.point_wise.1.weight', 'tpu_batch_normalization_2/gamma/ExponentialMovingAverage'),
        ('block.point_wise.1.bias', 'tpu_batch_normalization_2/beta/ExponentialMovingAverage'),
        ('block.point_wise.1.running_mean', 'tpu_batch_normalization_2/moving_mean/ExponentialMovingAverage'),
        ('block.point_wise.1.running_var', 'tpu_batch_normalization_2/moving_variance/ExponentialMovingAverage'),

        # head
        ('head.bottleneck.0.weight', 'head/conv2d/kernel/ExponentialMovingAverage'),
        ('head.bottleneck.1.weight', 'head/tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('head.bottleneck.1.bias', 'head/tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('head.bottleneck.1.running_mean', 'head/tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('head.bottleneck.1.running_var', 'head/tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # classifier
        ('head.classifier.weight', 'head/dense/kernel/ExponentialMovingAverage'),
        ('head.classifier.bias', 'head/dense/bias/ExponentialMovingAverage'),

        ('\\.(\\d+)\\.', lambda x: f'_{int(x.group(1))}/'),
    ]

    for name, param in list(model.named_parameters()) + list(model.named_buffers()):
        for pattern, sub in name_convertor:
            name = re.sub(pattern, sub, name)
        if 'dense/kernel' in name and list(param.shape) not in [[1000, 1280], [21843, 1280]]:
            continue
        if 'dense/bias' in name and list(param.shape) not in [[1000], [21843]]:
            continue
        if 'num_batches_tracked' in name:
            continue
        param.data.copy_(npz_dim_convertor(name, weight.get(name)))