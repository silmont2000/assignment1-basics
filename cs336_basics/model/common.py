import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from jaxtyping import Bool, Float, Int
from einops import einsum


def softmax(x: Tensor, i: int = -1):
    max_x = x.max(dim=i, keepdim=True).values
    numerator = Tensor.exp(x-max_x)
    denominator = Tensor.sum(numerator, dim=i, keepdim=True)
    return numerator/denominator


def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
                                 K: Float[Tensor, " ... keys d_k"],
                                 V: Float[Tensor, " ... values d_v"],
                                 mask: Bool[Tensor,
                                            " ... queries keys"] | None = None,
                                 ):
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / (Q.shape[-1] ** 0.5)
    minus_inf = -1e9  # 当负无穷使
    if mask is not None:
        scores = scores.masked_fill(mask == False, minus_inf)

    # 别忘了softmax！！！！
    weights = softmax(scores, i=-1)
    return einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
