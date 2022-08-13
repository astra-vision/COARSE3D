from __future__ import division

from bisect import bisect_right

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLR(_LRScheduler):
    """https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py"""

    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=0.1,
            warmup_steps=1,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_steps:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_steps
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class ClipLR(object):
    """Clip the learning rate of a given scheduler.
    Same interfaces of _LRScheduler should be implemented.

    Args:
        scheduler (_LRScheduler): an instance of _LRScheduler.
        min_lr (float): minimum learning rate.

    """

    def __init__(self, scheduler, min_lr=1e-5):
        assert isinstance(scheduler, _LRScheduler)
        self.scheduler = scheduler
        self.min_lr = min_lr

    def get_lr(self):
        return [max(self.min_lr, lr) for lr in self.scheduler.get_lr()]

    def __getattr__(self, item):
        if hasattr(self.scheduler, item):
            return getattr(self.scheduler, item)
        else:
            return getattr(self, item)


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step = self.global_step + 1
