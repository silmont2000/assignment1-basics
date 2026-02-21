from torch import Tensor
import torch
import torch.nn as nn
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.linear import Linear
from cs336_basics.model.common import softmax
from cs336_basics.model.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, theta, max_seq_len, device=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff,
                             theta, max_seq_len, device)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block.forward(x)

        x = self.final_norm(x)
        logits = self.final_linear(x)
        return logits
