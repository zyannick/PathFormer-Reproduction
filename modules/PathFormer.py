import torch
import torch.nn as nn
from modules.AdaptiveMultiScale import AdaptiveMultiScale

class PathFormer(nn.Module):
    
    def __init__(self, nb_layers):
        
        self.instance_norm = None
        self.AMSBlocks = nn.ModuleList()
        
        for _ in range(nb_layers):
            self.AMSBlocks.append(AdaptiveMultiScale())
            
        self.predictor = None
        
    def forward():
        
        pass