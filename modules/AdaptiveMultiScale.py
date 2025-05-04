import torch
from torch import nn
from modules.MultiScaleRouter import MultiScaleRouter

class AdaptiveMultiScale(nn.Module):
    
    def __init__(self, input_size, output_size, num_experts, device, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1, batch_norm=False):
        super(AdaptiveMultiScale, self).__init__()
        
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k
        
        self.multi_scale_router = MultiScaleRouter()
        self.patch_size_selector = None
        self.multi_scale_transformer_block = None
        self.multi_scale_aggregator = None
        
    def forward():
        pass