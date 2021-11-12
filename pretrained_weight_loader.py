import os
import subprocess
from pathlib import Path

import numpy as np


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
    model.load_tgz(load_tgz_from_url(url=model_urls[model_name], file_name=file_name))


def load_tgz_from_url(url, file_name):
    if not Path(file_name).exists():
        subprocess.run(["wget", "-r", "-nc", '-O', file_name, url])
    return np.load(file_name)