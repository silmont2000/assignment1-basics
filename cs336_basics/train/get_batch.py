import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    # 随机生成 batch_size 个起始索引
    # 最大索引不超过 len(dataset) - context_length - 1，确保 y 不越界
    # (batch_size,)意思是一维，（1,batch_size）就是二维了
    ix = torch.randint(high=len(dataset) - context_length,
                       size=(batch_size,))

    # 取x y
    # x 是从索引开始的 context_length 个词
    x_list = [torch.from_numpy(
        (dataset[i: i + context_length]).astype(np.int64)) for i in ix]
    # y 是 x 整体向右平移一位
    y_list = [torch.from_numpy(
        (dataset[i + 1: i + 1 + context_length]).astype(np.int64)) for i in ix]

    # 在 Batch 维度堆叠并推送到设备 (GPU/MPS/CPU)
    # 堆叠是为了不用for了，
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)

    return x, y
