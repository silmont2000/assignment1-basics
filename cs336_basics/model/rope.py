import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

# 这个确实一下子写不出性能好的版本，我只会遍历
# 所以这版本基本是让AI写的，特别是各种torch的用法


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.d_k = d_k

        denominator = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        numerator = torch.arange(max_seq_len).float()
        angle = torch.outer(numerator, denominator)

        # 每一列频率重复两次，变成 [f0, f0, f1, f1, f2, f2...]
        emb = torch.repeat_interleave(angle, 2, dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def rotate_half(self, x: Tensor) -> Tensor:
        """
        相邻配对置换：[x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        做 rotate_half 是为了用“向量加法”和“逐元素乘法”来实现原本复杂的“矩阵乘法”。
        """
        # RoPE Mathematical Logic:
        # Standard 2D rotation matrix:
        # [x1']   [ cosθ  -sinθ ] [x1]
        # [x2'] = [ sinθ   cosθ ] [x2]
        #
        # Expanded equations:
        # x1' = x1*cosθ - x2*sinθ
        # x2' = x1*sinθ + x2*cosθ
        #
        # Vectorized implementation used in forward():
        # [x1', x2'] = [x1, x2] * cosθ + [-x2, x1] * sinθ
        #               ^原向量 x,已经有了    ^旋转后的向量 rotate_half(x)，得算

        # 把 x 变成 (..., d_k/2, 2)
        # -1表示自动算，*x.shape是解包前面的维度。结果: 如果 x 原本是 [1, 2, 3, 4]，现在变成了 [[1, 2], [3, 4]]
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        # torch.unbind(dim): 这个函数会按指定的维度下刀劈开，并返回该维度上所有切片的元组
        # 比如[[1, 2], [3, 4]]，unbind(-1)会变成[1,3]和[2,4]。维度是先行后列，unbind(dim=0)就变成按行切开得到[1,2]和[3,4]
        x1, x2 = x_reshaped.unbind(dim=-1)

        # 按照 [-x2, x1] 堆叠并展平回去
        # dim=-1: 意思是“在最后再造一个维度把它们合回去”。
        # 结果: 刚才的 x1=[1, 3], x2=[2, 4] stack(dim=-1) 变成了在列上新增一个维度每个人放一列。 [[-2, 1], [-4, 3]]
        res = torch.stack((-x2, x1), dim=-1)
        return res.view(*x.shape)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: torch.Tensor) -> Float[Tensor, "... seq_len d_k"]:
        # 确保 cos/sin 长度足够并抓取对应位置
        cos = getattr(self, "cos_cached")[token_positions]
        sin = getattr(self, "sin_cached")[token_positions]

        # 执行旋转公式
        return (x * cos) + (self.rotate_half(x) * sin)
