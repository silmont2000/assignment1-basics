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
            d_model=d_model, device=device, dtype=dtype)
        self.RMSNorm_layer2 = RMSNorm(
            d_model=d_model, device=device, dtype=dtype)
        self.theta = theta
        self.multi_head_attention_layer = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.SwiGLU_layer = SwiGLU(
            d_model=d_model, d_ff=d_ff,  device=device, dtype=dtype)

    def forward(self, x: Tensor):
        rms1 = self.RMSNorm_layer1.forward(x)
        token_positions = None
        if self.theta > 0:
            t = x.shape[1]
            token_positions = torch.arange(t, device=x.device)
        attention = self.multi_head_attention_layer.forward(
            x=rms1, token_positions=token_positions)
        res1 = x + attention
        rms2 = self.RMSNorm_layer2.forward(res1)
        swiglu = self.SwiGLU_layer.forward(rms2)
        res2 = res1 + swiglu
        return res2
