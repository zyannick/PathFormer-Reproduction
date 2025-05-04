import torch
from torch import nn
from modules.layers.MultiScaleRouter import MultiScaleRouter
from modules.utils.SparseDispatcher import SparseDispatcher
from modules.layers.DualAttention import DualAttentionLayer
from ml_collections import ConfigDict

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=(1, 1),
                             bias=True)

    def forward(self, x):
        out = self.fc(x)
        return out

class AdaptiveMultiScale(nn.Module):

    def __init__(self, config: ConfigDict, layer_number: int):
        super(AdaptiveMultiScale, self).__init__()

        self.config = config

        self.num_experts = self.config.num_experts
        self.output_size = self.config.output_size
        self.input_size = self.config.input_size
        self.k = self.config.k
        self.d_model = self.config.d_model
        self.d_ff = self.config.d_ff
        self.dynamic = self.config.dynamic
        self.num_nodes = self.config.num_nodes
        self.layer_number = layer_number
        self.batch_norm = self.config.batch_norm
        self.residual_connection = self.config.residual_connection

        self.multi_scale_router = MultiScaleRouter(self.config)

        self.experts = nn.ModuleList()

        for patch in self.config.patch_size:
            patch_nums = int(self.input_size / patch)
            self.experts.append(
                DualAttentionLayer(
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dynamic=self.dynamic,
                    num_nodes=self.num_nodes,
                    patch_nums=patch_nums,
                    patch_size=patch,
                    factorized=True,
                    layer_number=self.layer_number,
                    batch_norm=self.batch_norm,
                )
            )
            
    def cv_squared(self, x: torch.Tensor):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, loss_coef=1e-2):

        gates, load = self.multi_scale_router(x, self.training)
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [
            self.experts[i](expert_inputs[i])[0] for i in range(self.num_experts)
        ]
        output = dispatcher.combine(expert_outputs)
        if self.residual_connection:
            output = output + x
        return output, balance_loss
