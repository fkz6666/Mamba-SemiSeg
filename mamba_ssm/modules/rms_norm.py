import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype=None):  # 显式声明参数
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x / torch.sqrt(norm + self.eps)
        return x_normed * self.weight



