import torch
from torch import nn


class AMSBlock(nn.Module):
    
    def __init__(self):
        self.multi_scale_router = None
        self.patch_size_selector = None
        self.multi_scale_transformer_block = None
        self.multi_scale_aggregator = None
        
    def forward():
        pass