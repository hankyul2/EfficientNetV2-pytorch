import math

from torch.optim.lr_scheduler import _LRScheduler


def compute_linear(x, a, b):
    return x * a + b


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, num_step: int, max_epochs: int, warmup_epoch: int = 1,
                 last_epoch: int = -1, verbose: bool = False):
        self.niter = max_epochs * num_step
        self.warmup = num_step * warmup_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self, step=None):
        step = step if step else self._step_count
        lrs = []
        for lr in self.base_lrs:
            if step < self.warmup:
                start_lr = lr / 5
                lrs.append(compute_linear(x=step / self.warmup, a=lr - start_lr, b=start_lr))
            else:
                cosine_y = (1 + math.cos(math.pi * (step - self.warmup) / (self.niter - self.warmup))) / 2
                lrs.append(compute_linear(x=cosine_y, a=lr, b=0))
        return lrs


class PowerLR(_LRScheduler):
    def __init__(self, optimizer, num_step: int, max_epochs: int, warmup_epoch: int = 1,
                 last_epoch: int = -1, verbose: bool = False):
        self.niter = max_epochs * num_step
        self.warmup = num_step * warmup_epoch
        super(PowerLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self, step=None):
        step = step if step else self._step_count
        lrs = []
        for lr in self.base_lrs:
            if step < self.warmup:
                start_lr = lr / 5
                lrs.append(compute_linear(x=step / self.warmup, a=lr - start_lr, b=start_lr))
            else:
                lrs.append(
                    compute_linear(x=(1 + 10 * (step - self.warmup) / (self.niter - self.warmup)) ** (-0.75), a=lr,
                                   b=0))
        return lrs


class FractionLR(_LRScheduler):
    def __init__(self, optimizer, num_step: int, warmup_epoch: int = 1, d_model: int = 512,
                 factor: int = 2, last_epoch: int = -1, verbose: bool = False):
        self.d_model = d_model
        self.factor = factor
        self.warmup = num_step * warmup_epoch
        super(FractionLR, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self, step=None):
        step = step if step else self._step_count
        lrs = []
        for lr in self.base_lrs:
            lrs.append(self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5)))
        return lrs