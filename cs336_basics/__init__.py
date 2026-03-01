import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

import torch

from .train.adamW import AdamW
from .transformer_lm import TransformerLM
from .train.common import cos_annealing_lr
from .train.get_batch import get_batch
from .train.common import cross_entropy, gradient_clipping
from .train.ckpt import save_checkpoint, load_checkpoint

from .pretokenization_example import find_chunk_boundaries
from .bpe.common import load_trained_bpe
from .bpe.bpe_tokenizer import bpe_tokenizer


config = {}

# config['optim_notes'] = "Fixed OOM: batch_size 256 调整到 128"

# model
config['vocab_size'] = 32000
config['d_model'] = 512
config['num_layers'] = 6
config['num_heads'] = 8
config['d_ff'] = 1344  # FFN层的中间隐藏层维度 (SwiGLU 标准推荐为 8/3 * d_model)
config['theta'] = 10000
config['max_seq_len'] = 256

# 优化器
config['lr'] = 6e-4
config['weight_decay'] = 0.01
config['betas'] = (0.9, 0.99)
config['eps'] = 1e-8

# 迭代
config['max_iters'] = 7001
# max_iters = 50
# 退火
config['warmup_iters'] = 70
config['cosine_cycle_iters'] = 7001
config['max_learning_rate'] = 6e-4
config['min_learning_rate'] = 1e-5
config['max_l2_norm'] = 1.0


# batch
config['batch_size'] = 32
config['gradient_accumulation_steps'] = 1  # 梯度累积步数，若 OOM 可调小 batch_size 并调大此参数
config['context_length'] = 256
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ckpt
save_interval = 500
eval_interval = 100
