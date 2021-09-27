# -*- coding: utf-8 -*-
"""
@Time   : 2021/9/22
@Author : Guikun Chen
"""
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, in_c, out_c):
        super(Baseline, self).__init__()
        self.layer = nn.Linear(in_c, out_c)

    def forward(self, data, device):
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1

        output = self.layer(flow_x)  # [B, N, Out_C], Out_C = D

        return output.unsqueeze(2)  # [B, N, 1, D=Out_C]
