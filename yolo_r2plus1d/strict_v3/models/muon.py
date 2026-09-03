"""Muon for Conv/Linear tensors, using a matrix view of N-D parameters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


@torch.no_grad()
def zeroth_power_newton_schulz(
    matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate the polar factor with the quintic Newton-Schulz iteration."""
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz input must be a matrix")
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.T if transposed else matrix
    x = x.to(torch.bfloat16)
    x /= x.norm().clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.T if transposed else x).to(matrix.dtype)


class NDimMuon(torch.optim.Optimizer):
    """Muon update for every parameter with at least two dimensions.

    Convolution kernels are viewed as ``(out_channels, -1)``. Bias and norm
    vectors should be optimized separately with AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.02,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if lr < 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("Invalid Muon hyperparameters")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                ns_steps=ns_steps,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim < 2:
                    raise ValueError("NDimMuon only accepts matrix-like parameters")
                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1.0 - group["momentum"])
                update = (
                    gradient.lerp(buffer, group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                matrix = update.reshape(update.shape[0], -1)
                orthogonal = zeroth_power_newton_schulz(
                    matrix, group["ns_steps"]
                ).reshape_as(update)
                adjusted_lr = group["lr"] * 0.2 * math.sqrt(max(matrix.shape))
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(orthogonal, alpha=-adjusted_lr)
        return loss
