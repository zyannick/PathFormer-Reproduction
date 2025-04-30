import torch
import torch.nn as nn
import torch.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce, repeat
import math


class SeasonalityBlock(nn.Module):
    
    def __init__(self, k = 1, prediction_length = 1, low_frequency = 1):
        self.k = k
        self.prediction_length = prediction_length
        self.low_frequency = low_frequency
        super(SeasonalityBlock, self).__init__()
        
    def dft(self, signal):
        dft_torch = fft.fft(signal)
        return dft_torch
    
    def topKFrequency(self, signal_freq):
        values, indices = torch.topk(signal_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(signal_freq.size(0)), torch.arange(signal_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        signal_freq = signal_freq[index_tuple]

        return signal_freq, index_tuple
    
    def idft(self, signal):
        idft_torch = fft.ifft(signal)
        return idft_torch
    
    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.prediction_length, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')
    
    def forward(self, signal):
        b, t, d = signal.shape
        
        signal_freq= fft.fft(signal, dim=1)
        
        if t % 2 == 0:
            signal_freq = signal_freq[:, self.low_frequency:-1]
            f = fft.rfftfreq(t)[self.low_frequency:-1]
        else:
            signal_freq = signal_freq[:, self.low_frequency:]
            f = fft.rfftfreq(t)[self.low_frequency:]
            
        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None
    
class TrendBlock(nn.Module):
    
    def __init__(self):
        super(TrendBlock, self).__init__()
    
    def forward(self, x):
        return x
    
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