import os
import torch
import numpy as np
import time
from cs336_basics import *


# model
vocab_size = 1000
d_model = 512
num_layers = 4
num_heads = 16
d_ff = 1344
theta = 10000
max_seq_len = 256

# 优化器
lr = 1e-3
weight_decay = 0.01
betas = (0.9, 0.999)
eps = 1e-8

# 迭代
# max_iters = 15000
max_iters = 50
# 退火
warmup_iters = max_iters*0.05
cosine_cycle_iters = max_iters
max_learning_rate = 8e-4
min_learning_rate = 8e-5
#
max_l2_norm = 1


# batch
batch_size = 256
context_length = 256
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ckpt
save_interval = 100
eval_interval = 100


def train_loop(train_data, val_data):
    model = TransformerLM(vocab_size, d_model, num_layers,
                          num_heads, d_ff, theta, max_seq_len, device)
    model = model.to(device)
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    start_time = time.time()

    for it in range(max_iters):
        model.train()
        # A. 更新学习率 (余弦退火)
        cur_lr = cos_annealing_lr(current_step=it, total_steps=cosine_cycle_iters,
                                  max_lr=max_learning_rate, min_lr=min_learning_rate, warmup_steps=warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

        # B. 获取 Batch 数据
        x, y = get_batch(dataset=train_data, batch_size=batch_size,
                         context_length=context_length, device=device)

        # C. 前向传播与损失计算
        logits = model.forward(x)
        loss = cross_entropy(inputs=logits, targets=y)

        # D. 反向传播与梯度裁剪
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 对应之前的 custom_grad_clipping
        total_norm = gradient_clipping(
            model.parameters(), max_l2_norm)  # type: ignore

        optimizer.step()

        if it % 10 == 0:
            print(
                f"Iter {it}: Loss {loss.item():.4f}, LR {cur_lr:.2e}, Norm {total_norm:.2f}")
        if it % eval_interval == 0:
            model.eval()  # 开启评估模式（关闭 Dropout 等）
            with torch.no_grad():  # 验证时不计算梯度，省显存
                vx, vy = get_batch(val_data, batch_size,
                                   context_length, device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
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
        tokenizer_dir, "TinyStoriesV2-GPT4-train-token.bin")
    valid_path = os.path.join(
        tokenizer_dir, "TinyStoriesV2-GPT4-valid-token.bin")

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(valid_path, dtype=np.uint16, mode="r")

    train_loop(train_data, val_data)
