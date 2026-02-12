import torch
from torch import Tensor
from cs336_basics.model.common import softmax
from jaxtyping import Bool, Float, Int
from einops import rearrange


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
