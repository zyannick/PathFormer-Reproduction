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
from torch.distributions.normal import Normal
from ml_collections import ConfigDict


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

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape

        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_frequency : -1]
            f = fft.rfftfreq(t)[self.low_frequency : -1]
        else:
            x_freq = x_freq[:, self.low_frequency :]
            f = fft.rfftfreq(t)[self.low_frequency :]

        x_freq, index_tuple = self.topKFrequency(x_freq)
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

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg_list:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1
        )
        res = x - moving_mean
        return res, moving_mean


class RoutingBlock(nn.Module):

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        top_k: int,
        noisy_gating: bool,
        num_nodes: int,
    ):
        super(RoutingBlock, self).__init__()
        if top_k > num_experts:
            raise ValueError("top_k (K) cannot be greater than num_experts (M)")

        self.num_experts = num_experts
        self.top_k = top_k

        self.noisy_gating = noisy_gating

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)

        self.w_gate = nn.Linear(input_size, num_experts)
        self.w_noise = nn.Linear(input_size, num_experts)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def _gates_to_load(self, gates: torch.Tensor):
        return (gates > 0).sum(0)

    def forward(self, x_trans: torch.Tensor, train: bool, noise_epsilon=1e-2):

        x = self.start_linear(x_trans).squeeze(-1)

        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.num_experts), dim=1
        )

        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.top_k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


class MultiScaleRouter(nn.Module):
    def __init__(self, config: ConfigDict, num_experts: int):
        super(MultiScaleRouter, self).__init__()
        self.config = config
        self.seasonality_block = SeasonalityBlock(
            self.config.k_seasonality, 0, self.config.low_frequency
        )
        self.trend_block = TrendBlock(self.config.list_kernel_size_trend)
        self.routing_block = RoutingBlock(
            self.config.seq_len,
            num_experts,
            self.config.k,
            self.config.noisy_gating,
            self.config.num_nodes,
        )

    def forward(self, x: torch.Tensor, train: bool):
        x = x[:, :, :, 0]
        _, trend = self.trend_block(x)
        seasonality, _ = self.seasonality_block(x)
        x_trans = x + seasonality + trend
        gates, load = self.routing_block(x_trans, train)
        return gates, load
