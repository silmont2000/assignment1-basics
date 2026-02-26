import os
import time

import numpy as np
import torch
import wandb

from cs336_basics import (
    AdamW,
    TransformerLM,
    config,
    cos_annealing_lr,
    cross_entropy,
    device,
    eval_interval,
    get_batch,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
    save_interval,
)


def debug_check(model, optimizer):
    p0 = next(model.parameters())
    print("param device:", p0.device, "dtype:", p0.dtype)

    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state[p]
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    print("state", k, "device:", v.device, "dtype:", v.dtype,
                          "max:", float(v.abs().max()))
            break
        break


def train_loop(train_data, val_data, cfg, resume_path: str | None = None):
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
    model = model.to(device)  # 在这to 统一修改所有param
    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=cfg["betas"],
        eps=cfg["eps"],
    )

    start_it = 0
    if resume_path is not None and os.path.exists(resume_path):
        start_it = 1 + load_checkpoint(
            resume_path, model, optimizer, device=device)

    start_time = time.time()
    p0 = next(model.parameters())
    debug_check(model, optimizer)

    grad_accum_steps = cfg.get("gradient_accumulation_steps", 1)
    optimizer.zero_grad(set_to_none=True)

    for it in range(start_it, cfg["max_iters"]):
        model.train()
        # A. 更新学习率 (余弦退火)
        cur_lr = cos_annealing_lr(
            current_step=it,
            total_steps=cfg["cosine_cycle_iters"],
            max_lr=cfg["max_learning_rate"],
            min_lr=cfg["min_learning_rate"],
            warmup_steps=cfg["warmup_iters"],
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

        # B. 梯度累积循环
        total_loss = 0
        for _ in range(grad_accum_steps):
            # 获取 Batch 数据
            x, y = get_batch(
                dataset=train_data,
                batch_size=cfg["batch_size"],
                context_length=cfg["context_length"],
                device=device,
            )

            # C. 前向传播与损失计算
            # 损失除以累积步数以保持量纲一致
            logits = model.forward(x, use_cache=True)['logits']
            loss = cross_entropy(inputs=logits, targets=y) / grad_accum_steps
            total_loss += loss.item()

            # D. 反向传播
            loss.backward()

        # E. 梯度裁剪与优化
        total_norm = gradient_clipping(
            model.parameters(), cfg["max_l2_norm"]
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        wandb.log(
            {
                "iter": it,
                "train/loss": total_loss,
                "lr": cur_lr,
                "grad_norm": total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm,
            },
            step=it,
        )

        if it % 10 == 0:
            print(
                f"Iter {it}: Loss {total_loss:.4f}, LR {cur_lr:.2e}, Norm {total_norm:.2f}")
        if it % eval_interval == 0:
            model.eval()  # 开启评估模式（关闭 Dropout 等）
            with torch.no_grad():  # 验证时不计算梯度，省显存
                vx, vy = get_batch(
                    val_data,
                    cfg["batch_size"],
                    cfg["context_length"],
                    device,
                )
                v_logits = model.forward(vx)['logits']
                v_loss = cross_entropy(v_logits, vy)
                wandb.log(
                    {
                        "iter": it,
                        "val/loss": v_loss.item(),
                    },
                    step=it,
                )
                print(f"--- Step {it}: Val Loss {v_loss.item():.4f} ---")
            model.train()  # 切回训练模式
        if it % save_interval == 0:
            save_checkpoint(model, optimizer, it, f"ckpt_step_{it}.pth")

    print(f"训练完成！总耗时: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    train_path = os.path.join(
        tokenizer_dir, "TinyStoriesV2-GPT4-train-token.bin"
    )
    valid_path = os.path.join(
        tokenizer_dir, "TinyStoriesV2-GPT4-valid-token.bin"
    )

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(valid_path, dtype=np.uint16, mode="r")

    # 注意：由于修改了模型架构（d_model 288 -> 512），旧的 checkpoint 无法直接加载。
    # 如果需要断点续传，请确保 checkpoint 对应的架构一致。
    resume_ckpt = None
    wandb.init(
        project="cs336-a1-transformer",
        name='TinyStories-6L-8H-512D-1000Steps',
        config=config
    )
    try:
        run_cfg = wandb.config
        train_loop(train_data, val_data, run_cfg, resume_path=resume_ckpt)
    finally:
        print("done!")
        wandb.finish()
