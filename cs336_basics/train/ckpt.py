import os
from typing import IO, Any, BinaryIO

import torch
from torch import Tensor


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes],):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    obj = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "iteration": iteration
    }
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,):
    obj = torch.load(src)
    optimizer.load_state_dict(obj['optimizer_state_dict'])
    model.load_state_dict(obj['model_state_dict'])
    # 获取进度
    iteration = obj['iteration']

    print(f"从步数 {iteration} 恢复")
    return iteration
