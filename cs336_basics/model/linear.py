import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        matrix = torch.empty((out_features, in_features), dtype=dtype)
        # 使用 GPT 风格的初始化：std=0.02
        torch.nn.init.trunc_normal_(matrix, std=0.02)
        self.W = nn.Parameter(matrix)

    def forward(self, x: torch.Tensor):
        result = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return result
