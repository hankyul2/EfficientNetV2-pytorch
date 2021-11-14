import math
import random
import numpy as np
import torch


def make_random_mask(H, W):
    ratio = np.random.beta(1.0, 1.0)
    h, w = int(math.sqrt(1 - ratio) * H), int(math.sqrt(1 - ratio) * W)
    row, col = random.randint(0, H - h), random.randint(0, W - w)
    mask = torch.ones((H, W))
    mask[row:row + h, col:col + w] = 0
    ratio = 1 - (h * w) / (H * W)
    return mask, ratio


def cutmix(x, y):
    B, C, H, W = x.shape
    mask, ratio = make_random_mask(H, W)
    mask, rand_idx = mask.to(x.device), torch.randperm(B).to(x.device)
    return mask * x + (1 - mask) * x[rand_idx], y, y[rand_idx], ratio


def cutout(x, y):
    B, C, H, W = x.shape
    mask, ratio = make_random_mask(H, W)
    return mask * x, y, ratio


def mixup(x, y):
    ratio = np.random.beta(1.0, 1.0)
    rand_idx = torch.randperm(x.size(0)).to(x.device)
    return ratio * x + (1 - ratio) * x[rand_idx], y, y[rand_idx], ratio
