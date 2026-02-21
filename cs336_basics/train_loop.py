import os
import wandb
import numpy as np
import time
import torch


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
    save_checkpoint,
    save_interval,
)


def train_loop(train_data, val_data, cfg):
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

    start_time = time.time()

    for it in range(cfg["max_iters"]):
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

        # B. 获取 Batch 数据
        x, y = get_batch(
            dataset=train_data,
            batch_size=cfg["batch_size"],
            context_length=cfg["context_length"],
            device=device,
        )

        # C. 前向传播与损失计算
        logits = model.forward(x)
        loss = cross_entropy(inputs=logits, targets=y)

        # D. 反向传播与梯度裁剪
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 对应之前的 custom_grad_clipping
        total_norm = gradient_clipping(
            model.parameters(), cfg["max_l2_norm"]
        )  # type: ignore

        optimizer.step()
        wandb.log(
            {
                "iter": it,
                "train/loss": loss.item(),
                "lr": cur_lr,
                "grad_norm": total_norm.item(),
            },
            step=it,
        )

        if it % 10 == 0:
            print(
                f"Iter {it}: Loss {loss.item():.4f}, LR {cur_lr:.2e}, Norm {total_norm:.2f}")
        if it % eval_interval == 0:
            model.eval()  # 开启评估模式（关闭 Dropout 等）
            with torch.no_grad():  # 验证时不计算梯度，省显存
                vx, vy = get_batch(
                    val_data,
                    cfg["batch_size"],
                    cfg["context_length"],
                    device,
                )
                v_logits = model(vx)
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
            # 对应图片第三点：序列化到指定路径
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

    wandb.init(
        project="cs336-a1-transformer",
        config=config,
    )
    try:
        run_cfg = wandb.config
        train_loop(train_data, val_data, run_cfg)
    finally:
        wandb.finish()
