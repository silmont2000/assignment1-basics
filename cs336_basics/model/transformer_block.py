from torch import Tensor
import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.swiglu import SwiGLU
from cs336_basics.model.multihead_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 0, max_seq_len: int = 0,   device=None, dtype=None):
        super().__init__()
        self.RMSNorm_layer1 = RMSNorm(
            d_model=d_model, device=None, dtype=dtype)
        self.RMSNorm_layer2 = RMSNorm(
            d_model=d_model, device=None, dtype=dtype)
        self.theta = theta
        self.multi_head_attention_layer = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.SwiGLU_layer = SwiGLU(
            d_model=d_model, d_ff=d_ff,  device=None, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None,
                kv_cache: tuple[Tensor, Tensor] | None = None,
                use_cache: bool = False):
        rms1 = self.RMSNorm_layer1.forward(x)
        if token_positions is None and self.theta > 0:
            t = x.shape[1]
            token_positions = torch.arange(t, device=x.device)

        mha_output = self.multi_head_attention_layer.forward(
            x=rms1, token_positions=token_positions, kv_cache=kv_cache, use_cache=use_cache)

        new_kv_cache = None
        if use_cache or kv_cache is not None:
            attention = mha_output['attention']
            new_kv_cache = mha_output['kv']
        else:
            attention = mha_output['attention']

        res1 = x + attention
        rms2 = self.RMSNorm_layer2.forward(res1)
        swiglu = self.SwiGLU_layer.forward(rms2)
        res2 = res1 + swiglu

        if use_cache or kv_cache is not None:
            return {"res2": res2, "new_kv_cache": new_kv_cache}
        return {"res2": res2, "new_kv_cache": None}
