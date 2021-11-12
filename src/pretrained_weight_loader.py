import os
import shutil
import subprocess
from pprint import pprint
from pathlib import Path

import numpy as np

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


def load_tgz(model, weight):
    for key, value in model.state_dict().items():
        print(key, value.shape)
    assert False


def load_tgz_from_url(url, file_name):
    if not Path(file_name).exists():
        subprocess.run(["wget", "-r", "-nc", '-O', file_name, url])
        shutil.unpack_archive(file_name)
    ckpt_path = os.path.splitext(file_name)[0]
    pretrained_ckpt = tf.train.latest_checkpoint(ckpt_path)
    np.save(f"{ckpt_path}.npy", {name:np.array(tf.train.load_variable(ckpt_path, name)) for name, shape in tf.train.list_variables(pretrained_ckpt)})
    return np.load(f"{ckpt_path}.npy", allow_pickle=True).item()