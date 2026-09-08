"""AdamW with FP32 master weights and moment states for BF16 models."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


class FP32AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float()
                if gradient.is_sparse:
                    raise RuntimeError("FP32AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["master"] = parameter.detach().float().clone()
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                elif "master" not in state:
                    # Legacy checkpoints did not retain master weights. Initialize
                    # from the current parameter once, then accumulate in FP32.
                    state["master"] = parameter.detach().float().clone()
                state["step"] += 1
                master = state["master"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = group["lr"] / bias_correction1
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                if group["weight_decay"]:
                    master.mul_(1.0 - group["lr"] * group["weight_decay"])
                master.addcdiv_(exp_avg, denominator, value=-step_size)
                parameter.copy_(master.to(parameter.dtype))
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            for key in ("master", "exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key].float()

    def assert_fp32_states(self) -> None:
        for state in self.state.values():
            for key in ("master", "exp_avg", "exp_avg_sq"):
                if key in state and state[key].dtype != torch.float32:
                    raise RuntimeError(f"AdamW state {key} is not FP32")
