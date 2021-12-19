from collections import Iterable

import torch
from torch import nn


def is_compute_layer(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


class NNComputeModuleProfile:
    def __init__(self, module, input_dim, output_dim):
        self.weight = module.weight.data
        self.num_param = module.weight.data.numel()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask = torch.IntTensor((self.weight != 0).int())
        self.is_conv = self.weight.dim() == 4
        self.numzeros = (self.mask == 0).sum().item()

    def save(self, path):
        profile = self.__dict__
        torch.save(profile, path)


class ProfilerResult:
    def __init__(self, macs=0, num_params=0, num_act=0, num_dp=0, per_compute_layer_complexity=()):
        # per_compute_layer complexity data format:
        # module name, num_op, num_params
        self.macs = macs
        self.num_params = num_params
        self.num_act = num_act
        self.num_dp = num_dp
        self.per_compute_layer_complexity = per_compute_layer_complexity
