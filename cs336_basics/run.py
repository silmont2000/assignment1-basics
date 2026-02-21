import os

import torch

from cs336_basics import (
    AdamW,
    TransformerLM,
    config,
    device,
    load_checkpoint,
    load_trained_bpe,
    bpe_tokenizer,
)

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
        shifted[..., 1:] = sorted_indices_to_remove[..., :-1]
        shifted[..., 0] = False
        sorted_indices_to_remove = shifted

    sorted_logits = sorted_logits.clone()
    sorted_logits[sorted_indices_to_remove] = -float("inf")
    filtered_logits = torch.full_like(logits, -float("inf"))
    filtered_logits.scatter_(0, sorted_indices, sorted_logits)
    return filtered_logits


def sample_from(logits):
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


if __name__ == "__main__":
    cfg = config
    model = TransformerLM(
        cfg["vocab_size"],
        cfg["d_model"],
        cfg["num_layers"],
        cfg["num_heads"],
        cfg["d_ff"],
        cfg["theta"],
        cfg["max_seq_len"],
        device,
    )
    model = model.to(device)
    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=cfg["betas"],
        eps=cfg["eps"],
    )

    base_dir = os.path.dirname(os.path.dirname(__file__))
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    pkl_path = os.path.join(tokenizer_dir, "tokenizer_tinystory.pkl")

    data = load_trained_bpe(pkl_path)
    tokenizer = bpe_tokenizer(
        vocab=data["vocab"],
        merges=data["merges"],
        special_tokens=["<|endoftext|>"],
    )
    ckpt_path = os.path.join(base_dir, "ckpt_step_200.pth")
    load_checkpoint(
        '/Users/xieboyang/Desktop/Robotic/CS36/ckpt_step_500.pth', model, optimizer)

    prompt = "Once upon a time there was a little boy named Ben."
    output = decode(prompt, 200, 0.8, 0.9, tokenizer, model)
    print("Prompt:", prompt)
    print("Output:", output)
