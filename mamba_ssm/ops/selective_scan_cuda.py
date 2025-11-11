# selective_scan_cuda.py
# Stub for selective_scan_cuda to avoid import error in non-CUDA environments.

import torch

def selective_scan_fwd(u, k, v, x, delta, delta_bias=None, delta_softplus=False):
    # 模拟输出：直接返回 v + x（仅做占位）
    y = v + x
    return y, torch.zeros_like(y)

def selective_scan_bwd(dy, u, k, v, x, delta, y, cache):
    # 模拟反向传播结果：全部返回零梯度
    return (
        torch.zeros_like(u),
        torch.zeros_like(k),
        torch.zeros_like(v),
        torch.zeros_like(x),
        torch.zeros_like(delta),
        None  # delta_bias 梯度
    )
