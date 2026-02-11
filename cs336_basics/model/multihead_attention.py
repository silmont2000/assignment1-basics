import torch
import torch.nn as nn
from einops import einsum, rearrange
from cs336_basics.model.linear import Linear
from cs336_basics.model.common import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.WK = Linear(d_model, d_model)
        self.WQ = Linear(d_model, d_model)
        self.WV = Linear(d_model, d_model)
        self.WO = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        # fmt: off
        K = rearrange(self.WK.forward(x), '... s (h d) -> ... h s d', h=self.num_heads)
        Q = rearrange(self.WQ.forward(x), '... s (h d) -> ... h s d', h=self.num_heads)
        V = rearrange(self.WV.forward(x), '... s (h d) -> ... h s d', h=self.num_heads)
        # fmt: on

        batch, seq_len, _ = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        attention = scaled_dot_product_attention(Q, K, V, mask=mask)
        combined_attention = rearrange(attention, '... h s d -> ... s (h d)')

        return self.WO.forward(combined_attention)
