import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
from cs336_basics.model.linear import Linear
from cs336_basics.model.common import scaled_dot_product_attention
from jaxtyping import Bool, Float, Int

from cs336_basics.model.rope import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = 0, max_seq_len: int = 0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.WK = Linear(d_model, d_model)
        self.WQ = Linear(d_model, d_model)
        self.WV = Linear(d_model, d_model)
        self.WO = Linear(d_model, d_model)
        if theta > 0:
            self.RoPE_layer = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_model//self.num_heads, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: Int[Tensor, " ... sequence_length"] | None = None):


        # fmt: off
        K = rearrange(self.WK.forward(x), '... s (h d) -> ... h s d', h=self.num_heads)
        Q = rearrange(self.WQ.forward(x), '... s (h d) -> ... h s d', h=self.num_heads)
        V = rearrange(self.WV.forward(x), '... s (h d) -> ... h s d', h=self.num_heads)
        # fmt: on
        batch, seq_len, _ = x.shape

        if token_positions is not None and self.RoPE_layer:
            K = self.RoPE_layer.forward(
                x=K, token_positions=token_positions)
            Q = self.RoPE_layer.forward(
                x=Q, token_positions=token_positions)
            # V 不转！！！！
            # V = self.RoPE_layer.forward(
            # x=V, token_positions=token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        attention = scaled_dot_product_attention(Q, K, V, mask=mask)
        combined_attention = rearrange(attention, '... h s d -> ... s (h d)')

        return self.WO.forward(combined_attention)
