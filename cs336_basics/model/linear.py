import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        matrix = torch.empty((in_features, out_features),
                             device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(matrix)
        self.W = nn.Parameter(matrix)

    def forward(self, x: torch.Tensor):
        return einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")

    def forward_with_w(self, x: torch.Tensor, w: torch.Tensor):
        return einsum(x, w, "... d_in, d_out d_in -> ... d_out")
