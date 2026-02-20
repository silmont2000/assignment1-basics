from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, ε=1e-8, β1=0.99, β2=0.999, λ=1e-2):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= β1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {β1}")
        if not 0.0 <= β2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {β2}")
        defaults = {"lr": lr, "ε": ε, "β1": β1, "β2": β2, "λ": λ}
        super().__init__(params, defaults)
        # param_groups是超参
        # state是要优化的变量，Parameter
    # fmt: off
    def step(self, closure: Optional[Callable] = None) -> Optional[float]: # type: ignore
    # fmt: on
        loss = None if closure is None else closure()
        for group in self.param_groups:
            ε = group["ε"]
            β1 = group["β1"]
            β2 = group["β2"]
            λ = group["λ"]
            lr = group["lr"]  # 拿到当前这一组的学习率

            for p in group["params"]:  # 然后遍历这组的参数，全部用这个学习旅更新
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    # 只初始化一次，get的话是每次get都要白白占用

                # AdamW 直接作用于参数
                p.data.mul_(1 - lr * λ)
                # 得到时间步
                t = state.get("t", 1)
                # 动量
                mt_1 = state['m']
                vt_1 = state['v']
                # 得到这轮迭代的梯度
                grad = p.grad.data
                # 梯度更新，同时学习率随步数衰减
                # mt = β1*mt_1+(1-β1)*grad
                # state["m"] = mt
                mt_1.mul_(β1).add_(grad, alpha=1 - β1)  # 这样更省内存

                # vt = β2*vt_1+(1-β2)*grad*grad
                # state["v"] = vt
                vt_1.mul_(β2).addcmul_(grad, grad, value=1 - β2)

                bias_correction1 = 1 - β1 ** t
                bias_correction2 = 1 - β2 ** t
                step_size = lr / bias_correction1
                denom = (vt_1.sqrt() / math.sqrt(bias_correction2)).add_(ε)
                p.data.addcdiv_(mt_1, denom, value=-step_size)
                # mt = mt_1/(1-β1**t)
                # vt = vt_1/(1-β2**t)
                # update = mt/(vt**0.5+ε)
                # p.data -= lr*(update+λ*p.data)
                # 作用于学习率更省性能

                state["t"] = t + 1  # 时间步更新
        return loss
