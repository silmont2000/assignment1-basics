import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        # self.device = device
        # self.dtype = dtype

        matrix = torch.empty((1, d_model),
                             device=device, dtype=dtype)
        # matrix = torch.empty((num_embeddings, embedding_dim),
        #                      device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(matrix)
        self.g = nn.Parameter(matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # 防止平方越界的
        # (batch_size, sequence_length, d_model)
        tmp = einsum(x, x, "... d_model, ... d_model -> ...")
        tmp = tmp/self.d_model + self.eps
        rms = torch.sqrt(tmp).unsqueeze(-1)
        result = x/rms*self.g
        return result.to(in_dtype)

    def forward_with_w(self, x: torch.Tensor, g: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        tmp = einsum(x, x, "... d_model, ... d_model -> ...")
        tmp = tmp/self.d_model + self.eps
        rms = torch.sqrt(tmp).unsqueeze(-1)
        result = x/rms*g
        return result.to(in_dtype)
