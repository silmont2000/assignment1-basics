import torch
from torch import Tensor
from cs336_basics.model.common import softmax
from jaxtyping import Bool, Float, Int


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
