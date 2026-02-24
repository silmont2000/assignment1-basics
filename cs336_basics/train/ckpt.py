import os
from typing import IO, Any, BinaryIO

import torch
from torch import Tensor


class TrainingDivergedError(Exception):
    """当模型 Loss 或权重出现 NaN 时抛出的自定义异常"""
    pass


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
                    optimizer: torch.optim.Optimizer,
                    device: str | torch.device | None = None):
    obj = torch.load(src, map_location=device)
    # obj = torch.load(src)
    for name, param in obj['model_state_dict'].items():
        if torch.isnan(param).any():
            raise TrainingDivergedError(
                f"警告：权重 {name} 中已经包含 NaN 了！这个 ckpt 是坏的")
        if torch.max(torch.abs(param)) > 1e4:
            print(f"警告：权重 {name} 数值过大 ({torch.max(param)})，易崩")

    optimizer.load_state_dict(obj['optimizer_state_dict'])
    model.load_state_dict(obj['model_state_dict'])
    # 获取进度
    iteration = obj['iteration']

    print(f"从步数 {iteration} 恢复完成，已加载模型权重与优化器状态 (t, m, v)")
    return iteration
