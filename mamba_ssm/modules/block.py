# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional
from functools import partial

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
from mamba_ssm.modules.rms_norm import RMSNorm


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        mlp_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is for performance reasons, to enable fused add and norm.
        """
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # 统一处理 norm 构造器
        norm_cls_base = norm_cls.func if isinstance(norm_cls, partial) else norm_cls
        factory_kwargs = {"device": self.device, "dtype": self.dtype}
        factory_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}

        # 初始化 norm
        self.norm = norm_cls(dim, **factory_kwargs)

        # ✅ 添加缺失的 mixer 实例化
        self.mixer = mixer_cls(dim)

        # 初始化 MLP 和第二层 norm（如有）
        if mlp_cls is not nn.Identity:
            if norm_cls_base is RMSNorm:
                self.norm2 = norm_cls(dim)
            else:
                self.norm2 = norm_cls(dim, **factory_kwargs)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import failed"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
