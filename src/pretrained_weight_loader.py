import os
import re
import subprocess
from pathlib import Path

import numpy as np
from einops import rearrange

import torch


model_urls = {
    "efficientnet_v2_s": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s.tgz",
    "efficientnet_v2_m": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz",
    "efficientnet_v2_l": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-l.tgz",
    "efficientnet_v2_s_in21k": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s-21k.tgz",
    "efficientnet_v2_m_in21k": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m-21k.tgz",
    "efficientnet_v2_l_in21k": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-l-21k.tgz",
    "efficientnet_v2_xl_in21k": "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-xl-21k.tgz",
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
        if weight.shape[3] == 1:
            # depth-wise convolution
            weight = rearrange(weight, 'h w in_c out_c -> in_c out_c h w')
        else:
            weight = rearrange(weight, 'h w in_c out_c -> out_c in_c h w')
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
        ('head.0.weight', 'head/conv2d/kernel/ExponentialMovingAverage'),
        ('head.1.weight', 'head/tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('head.1.bias', 'head/tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('head.1.running_mean', 'head/tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('head.1.running_var', 'head/tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        ('\\.(\\d+)\\.', lambda x: f'_{int(x.group(1))}/'),
    ]

    for name, param in model.named_parameters():
        for pattern, sub in name_convertor:
            name = re.sub(pattern, sub, name)
        param.data.copy_(npz_dim_convertor(name, weight.get(name)))