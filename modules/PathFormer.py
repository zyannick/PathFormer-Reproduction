import torch
import torch.nn as nn
from ml_collections import ConfigDict
from modules.layers.RevIN import RevIN
from modules.layers.AdaptiveMultiScale import AdaptiveMultiScale


class PathFormer(nn.Module):

    def __init__(self, config: ConfigDict):
        super(PathFormer, self).__init__()
        self.config = config
        self.layer_nums = self.config.layer_nums
        self.revin = self.config.revin
        if self.revin:
            self.revin_layer = RevIN(
                num_features=self.config.num_nodes, affine=False, subtract_last=False
            )

        self.start_fc = nn.Linear(in_features=1, out_features=self.config.d_model)
        self.AMS_lists = nn.ModuleList()

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AdaptiveMultiScale(
                    config=self.config,
                    patch_size=self.config.patch_size_list[num],
                    num_experts=self.config.num_experts_list[num],
                    layer_number=num + 1,
                )
            )
        self.projections = nn.Sequential(
            nn.Linear(self.config.seq_len * self.config.d_model, self.config.pred_len)
        )

    def forward(self, x):

        x = x.to(torch.float32)

        balance_loss = 0
        if self.revin:
            x = self.revin_layer(x, "norm")
        out = self.start_fc(x.unsqueeze(-1))

        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.config.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        if self.revin:
            out = self.revin_layer(out, "denorm")

        return {"prediction": out, "balance_loss": balance_loss}
