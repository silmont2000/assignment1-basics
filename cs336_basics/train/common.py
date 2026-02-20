import torch
from torch import Tensor
from cs336_basics.model.common import softmax
from jaxtyping import Bool, Float, Int
from einops import rearrange
import math


def cross_entropy(inputs: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."]):
    vocab_size = inputs.shape[-1]
    flattened_inputs = rearrange(inputs, '... vocab_size -> (...) vocab_size')
    flattened_targets = rearrange(targets, '... -> (...) 1')

    # max: Float[Tensor, " ... vocab_size"]
    max, _ = flattened_inputs.max(dim=-1, keepdim=True)
    shifted_inputs = flattened_inputs-max
    exp_inputs = torch.exp(shifted_inputs)
    log_sum_exp = torch.log(torch.sum(exp_inputs, dim=-1, keepdim=True))

    predicted_value = torch.gather(
        dim=-1, input=shifted_inputs, index=flattened_targets)

    loss = log_sum_exp-predicted_value

    return loss.mean()


def cos_annealing_lr(current_step, total_steps, max_lr, min_lr, warmup_steps):

    # Warm-up
    if current_step < warmup_steps:
        return (current_step / warmup_steps) * max_lr

    # Post-annealing
    if current_step > total_steps:
        return min_lr

    # Cosine annealing
    # 计算在余弦曲线中的进度 (0 到 1 之间)
    decay_ratio = (current_step - warmup_steps) / (total_steps - warmup_steps)

    # 余弦公式计算
    # math.cos(math.pi * decay_ratio) 的范围是 [1, -1]
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def gradient_clipping(parameters, max_l2_norm):
    grads = [p.grad.detach().flatten()
             for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.0)

    # 将所有梯度拼接成一个长向量，然后计算范数
    total_norm = torch.cat(grads).norm(2)
    clip_coeff = max_l2_norm / (total_norm + 1e-10)
    if clip_coeff < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coeff)

    return total_norm
