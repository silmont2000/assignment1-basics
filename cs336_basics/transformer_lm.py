from torch import Tensor
import torch
import torch.nn as nn
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.common import softmax
from cs336_basics.model.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, theta, max_seq_len, device=None):
        super().__init__()

        self.token_embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff,
                             theta, max_seq_len, device)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

    def forward(self, input_ids, kv_caches: list[tuple[Tensor, Tensor]] | None = None,
                use_cache: bool = False):
        x = self.token_embedding.forward(token_ids=input_ids)

        # Handle token positions for RoPE
        batch_size, seq_len = input_ids.shape
        # kv_caches 是一个列表，最外层长度为 num_layers，每一项对应 Transformer 中的一个 Block
        # 中间层存的是一个元组(Key_Tensor, Value_Tensor).
        # K或者V都是[Batch_Size, Num_Heads, Sequence_Length, _]
        if kv_caches is not None:
            # 已有的Sequence_Length
            past_len = kv_caches[0][0].shape[-2]
        else:
            past_len = 0
        # 把这次要填充的位置id算出来
        token_positions = torch.arange(
            past_len, past_len + seq_len, device=input_ids.device)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            # 旧kvcache
            kv_cache = kv_caches[i] if kv_caches is not None else None
            block_output = block.forward(x, token_positions=token_positions,
                                         kv_cache=kv_cache, use_cache=use_cache)

            if use_cache or kv_cache is not None:
                x = block_output['res2']
                new_kv = block_output['new_kv_cache']
                new_kv_caches.append(new_kv)  # 增加上
            else:
                x = block_output['res2']

        x = self.final_norm(x)
        logits = self.final_linear(x)

        if use_cache or kv_caches is not None:
            return {'logits': logits, "new_kv_caches": new_kv_caches}
        return {'logits': logits, "new_kv_caches": None}
