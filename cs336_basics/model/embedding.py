import torch
import torch.nn as nn
from einops import einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        matrix = torch.empty((num_embeddings, embedding_dim),
                             device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(matrix)
        self.embedding = nn.Parameter(matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
