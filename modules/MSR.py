import torch
import torch.nn as nn
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce, repeat
import math
from typing import Optional, Tuple, Union, List, Dict
from torch import Tensor
from torch.nn import functional as F

class SeasonalityBlock(nn.Module):

    def __init__(
        self, k: int = 1, prediction_length: int = 1, low_frequency: float = 1
    ):
        super(SeasonalityBlock, self).__init__()
        self.k = k
        self.prediction_length = prediction_length
        self.low_frequency = low_frequency

    def topKFrequency(self, signal_freq: torch.Tensor):
        values, indices = torch.topk(
            signal_freq.abs(), self.k, dim=1, largest=True, sorted=True
        )
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(signal_freq.size(0)), torch.arange(signal_freq.size(2))
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        signal_freq = signal_freq[index_tuple]

        return signal_freq, index_tuple

    def extrapolate(self, x_freq: torch.Tensor, f: torch.Tensor, t: int):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(
            torch.arange(t + self.prediction_length, dtype=torch.float),
            "t -> () () t ()",
        ).to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, "b f d -> b f () d")
        phase = rearrange(x_freq.angle(), "b f d -> b f () d")

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, "b f t d -> b t d", "sum")

    def forward(self, signal: torch.Tensor):
        b, t, d = signal.shape

        signal_freq = fft.fft(signal, dim=1)

        if t % 2 == 0:
            signal_freq = signal_freq[:, self.low_frequency : -1]
            f = fft.rfftfreq(t)[self.low_frequency : -1]
        else:
            signal_freq = signal_freq[:, self.low_frequency :]
            f = fft.rfftfreq(t)[self.low_frequency :]

        x_freq, index_tuple = self.topKFrequency(signal_freq)
        f = repeat(f, "f -> b f d", b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], "b f d -> b f () d").to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None


class TrendBlock(nn.Module):

    def __init__(self, list_kernels : List[int] = [1, 2, 3, 4, 5]):
        super(TrendBlock, self).__init__()
        self.list_kernels = list_kernels
        
        self.list_averages_poolings = nn.ModuleList()
        
        for kernel in self.list_kernels:
            self.list_averages_poolings.append(
                nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2)
            )
            
        self.weight = nn.Parameter(torch.ones(len(self.list_kernels))) 

    def forward(self, x):
        pools = []
        for i, kernel in enumerate(self.list_kernels):
            pools.append(self.list_averages_poolings[i](x))
        
        pooled_stack = torch.stack(pools, dim=0)  
        weight = F.softmax(self.weight, dim=0).view(-1, 1, 1, 1)
        weighted_sum = (pooled_stack * weight).sum(dim=0)  

        output = weighted_sum + x
        return output


class RoutingBlock(nn.Module):

    def __init__(self):
        super(RoutingBlock, self).__init__()

    def forward(self, x):
        return x


class MultiScaleRouter(nn.Module):

    def __init__(self):
        super(MultiScaleRouter, self).__init__()

    def forward(self, x):
        pass
