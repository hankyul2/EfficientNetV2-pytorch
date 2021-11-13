import os
import re
import shutil
import subprocess
from pprint import pprint
from pathlib import Path

from einops import rearrange
import numpy as np

import torch
import tensorflow as tf


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
    load_tgz(model, load_tgz_from_url(url=model_urls[model_name], file_name=file_name))


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


def load_tgz(model, weight):
    # for k, v in model.named_parameters():
    #     print(k, v.shape)
    for k, v in weight.items():
        if 'ExponentialMovingAverage' not in k:
            print(k, v.shape)

    name_convertor = [
        # stem
        ('stem.0.weight', 'stem/conv2d/kernel'),
        ('stem.1.weight', 'stem/tpu_batch_normalization/beta'),
        ('stem.1.bias', 'stem/tpu_batch_normalization/gamma'),
        ('stem.1.running_mean', 'stem/tpu_batch_normalization/moving_mean'),
        ('stem.1.running_var', 'stem/tpu_batch_normalization/moving_variance'),

        # fused layer
        ('block.fused.0.weight', 'conv2d/kernel'),
        ('block.fused.1.weight', 'tpu_batch_normalization/beta'),
        ('block.fused.1.bias', 'tpu_batch_normalization/gamma'),
        ('block.fused.1.running_mean', 'tpu_batch_normalization/moving_mean'),
        ('block.fused.1.running_var', 'tpu_batch_normalization/moving_variance'),

        # linear bottleneck
        ('block.linear_bottleneck.0.weight', 'conv2d/kernel'),
        ('block.linear_bottleneck.1.weight', 'tpu_batch_normalization/beta'),
        ('block.linear_bottleneck.1.bias', 'tpu_batch_normalization/gamma'),
        ('block.linear_bottleneck.1.running_mean', 'tpu_batch_normalization/moving_mean'),
        ('block.linear_bottleneck.1.running_var', 'tpu_batch_normalization/moving_variance'),

        # depth wise layer
        ('block.depth_wise.0.weight', 'depthwise_conv2d/depthwise_kernel'),
        ('block.depth_wise.1.weight', 'tpu_batch_normalization_1/beta'),
        ('block.depth_wise.1.bias', 'tpu_batch_normalization_1/gamma'),
        ('block.depth_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean'),
        ('block.depth_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance'),

        # se layer
        ('block.se.fc1.weight', 'se/conv2d/kernel'), ('block.se.fc1.bias', 'se/conv2d/bias'),
        ('block.se.fc2.weight', 'se/conv2d_1/kernel'), ('block.se.fc2.bias', 'se/conv2d_1/bias'),

        # point wise layer
        ('block.fused_point_wise.0.weight', 'conv2d_1/kernel'),
        ('block.fused_point_wise.1.weight', 'tpu_batch_normalization_1/beta'),
        ('block.fused_point_wise.1.bias', 'tpu_batch_normalization_1/gamma'),
        ('block.fused_point_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean'),
        ('block.fused_point_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance'),

        ('block.point_wise.0.weight', 'conv2d_1/kernel'),
        ('block.point_wise.1.weight', 'tpu_batch_normalization_2/beta'),
        ('block.point_wise.1.bias', 'tpu_batch_normalization_2/gamma'),
        ('block.point_wise.1.running_mean', 'tpu_batch_normalization_2/moving_mean'),
        ('block.point_wise.1.running_var', 'tpu_batch_normalization_2/moving_variance'),

        # head
        ('head.0.weight', 'head/conv2d/kernel'),
        ('head.1.weight', 'head/tpu_batch_normalization/beta'),
        ('head.1.bias', 'head/tpu_batch_normalization/gamma'),
        ('head.1.running_mean', 'head/tpu_batch_normalization/moving_mean'),
        ('head.1.running_var', 'head/tpu_batch_normalization/moving_variance'),

        ('\\.(\\d+)\\.', lambda x: f'_{int(x.group(1))}/'),
    ]
    for name, param in model.named_parameters():
        bc_name = name
        for pattern, sub in name_convertor:
            name = re.sub(pattern, sub, name)
        print(bc_name, '->', name)
        print(param.shape)
        param.data.copy_(npz_dim_convertor(name, weight.get(name)))

    assert False


def load_tgz_from_url(url, file_name):
    ckpt_path = os.path.splitext(file_name)[0]
    if not Path(file_name).exists():
        subprocess.run(["wget", "-r", "-nc", '-O', file_name, url])
        shutil.unpack_archive(file_name, os.path.dirname(file_name))
    pretrained_ckpt = tf.train.latest_checkpoint(ckpt_path)
    np.save(f"{ckpt_path}.npy", {'/'.join(name.split('/')[1:]):np.array(tf.train.load_variable(ckpt_path, name)) for name, shape in tf.train.list_variables(pretrained_ckpt)})
    return np.load(f"{ckpt_path}.npy", allow_pickle=True).item()