import os
import torch
import numpy as np
import time
from cs336_basics import *

EOS_TOKEN = b"<|endoftext|>"


def decode(prompt, max_tokens, temperature, top_p, tokenizer: bpe_tokenizer, model: TransformerLM):
    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.reverse_vocab[EOS_TOKEN]
    device = next(model.parameters()).device

    for _ in range(max_tokens):
        input_ids = torch.tensor(tokens, dtype=torch.long,
                                 device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model.forward(input_ids)
        next_token_logits = logits[0, -1, :]
        next_token_logits = next_token_logits / temperature

        filtered_logits = top_p_filter(next_token_logits, top_p)
        next_token = sample_from(filtered_logits)

        if next_token == eos_id:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)


def top_p_filter(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    if sorted_indices_to_remove.any():
        shifted = sorted_indices_to_remove.clone()
        shifted[..., 1:] = shifted[..., :-1]
        shifted[..., 0] = False
        sorted_indices_to_remove = shifted

    sorted_logits[sorted_indices_to_remove] = -float("inf")
    filtered_logits = torch.full_like(logits, -float("inf"))
    filtered_logits.scatter_(0, sorted_indices, sorted_logits)
    return filtered_logits


def sample_from(logits):
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
