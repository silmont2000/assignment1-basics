import torch
import torch.nn as nn
from einops import einsum
from torch import Tensor
from jaxtyping import Bool, Float, Int
from math import ceil


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = -1,
                 device=None, dtype=None,):
        super().__init__()
        if d_ff == -1:
            d_ff = int(ceil(8./3. * d_model/64)*64)
        self.d_ff = d_ff

        self.W1 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype))
        self.W2 = nn.Parameter(torch.empty((d_model, d_ff), dtype=dtype))
        self.W3 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype))

        for p in [self.W1, self.W2, self.W3]:
            torch.nn.init.trunc_normal_(p)

    def forward(self, x:  Float[Tensor, " ... d_model"]):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x1 = einsum(self.W1, x, "d_ff d_model, ... d_model -> d_ff ...")
        x2 = einsum(self.W3, x, "d_ff d_model, ... d_model -> d_ff ...")
        result = einsum(self.W2, SiLU(x1) * x2,
                        " d_model d_ff , d_ff ... -> ... d_model")

        return result.to(in_dtype)
