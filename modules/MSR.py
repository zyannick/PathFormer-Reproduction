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


class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1
        )
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TrendBlock(nn.Module):
    def __init__(self, kernel_size):
        super(TrendBlock, self).__init__()
        self.moving_avg_list = nn.ModuleList()
        for k in kernel_size:
            self.moving_avg_list.append(MovingAverage(k, 1))
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x_rem):
        moving_mean = []
        for func in self.moving_avg_list:
            moving_avg = func(x_rem)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        x_trend = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x_rem.unsqueeze(-1))), dim=-1
        )
        return x_trend


class RoutingBlock(nn.Module):

    def __init__(self, d: int, num_patch_sizes: int, top_k: int):
        super(RoutingBlock, self).__init__()
        if top_k > num_patch_sizes:
            raise ValueError("top_k (K) cannot be greater than num_patch_sizes (M)")

        self.d = d
        self.num_patch_sizes = num_patch_sizes
        self.top_k = top_k

        self.fc_r = nn.Linear(d, num_patch_sizes)
        self.fc_noise = nn.Linear(d, num_patch_sizes)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_trans):
        base_score = self.fc_r(x_trans)

        noise_base = self.fc_noise(x_trans)
        noise_softplus = self.softplus(noise_base)
        noise = torch.randn_like(noise_softplus)
        noise_term = noise * noise_softplus
        raw_weights = base_score + noise_term
        pathway_weights = self.softmax(raw_weights)

        top_k_values, top_k_indices = torch.topk(pathway_weights, self.top_k, dim=-1)

        sparse_weights = torch.zeros_like(pathway_weights)
        sparse_weights.scatter_(1, top_k_indices, top_k_values)

        return sparse_weights


class MultiScaleRouter(nn.Module):

    def __init__(
        self,
        top_k: int,
        prediction_length: int,
        low_frequency: float,
        list_kernel_size: List[int],
        num_patch_sizes: int,
        d: int,
    ):
        super(MultiScaleRouter, self).__init__()
        self.seasonality_block = SeasonalityBlock(
            top_k, prediction_length, low_frequency
        )
        self.trend_block = TrendBlock(list_kernel_size)
        self.linear_x_trans = nn.Linear()
        self.routing_block = RoutingBlock(d, num_patch_sizes, top_k)

    def forward(self, x):
        x = x[:, :, :, 0]
        x_seasonality = self.seasonality_block(x)
        x_trend = self.trend_block(x - x_seasonality)

        x_trans = self.linear_x_trans(x + x_seasonality + x_trend)

        r_trans = self.routing_block(x_trans)

        return r_trans
