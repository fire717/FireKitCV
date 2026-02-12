from typing import Callable, Optional, overload

import torch
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    def __init__(self, params: list[torch.Tensor], lr: float, momentum: float, weight_decay: float):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.weight_decay = weight_decay

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)  # whiten the update

                if self.weight_decay != 0:
                    p.data.mul_(1 - lr * self.weight_decay)  # apply weight decay

                p.data.add_(update, alpha=-lr)  # take a step
        return loss


class AdamMuon(Optimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        muon_momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
    ):
        muon_params, adam_params = [], []
        if isinstance(params, (list, tuple)) and isinstance(params[0], dict):
            for group in params:
                for p in group["params"]:
                    if p.ndim > 1:
                        muon_params.append(p)
                    else:
                        adam_params.append(p)
        else:
            for p in params:
                if p.ndim > 1:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        self.muon = Muon(muon_params, lr=lr, momentum=muon_momentum, weight_decay=weight_decay) if muon_params else None
        self.adam = AdamW(adam_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) if adam_params else None

        # Combine param groups for compatibility with PyTorch schedulers
        param_groups = []
        if self.muon:
            for g in self.muon.param_groups:
                g["optimizer"] = "muon"
                param_groups.append(g)
        if self.adam:
            for g in self.adam.param_groups:
                g["optimizer"] = "adam"
                param_groups.append(g)

        defaults = dict(lr=lr, momentum=muon_momentum, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        if self.muon:
            self.muon.step()
        if self.adam:
            self.adam.step()

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adam:
            self.adam.zero_grad(set_to_none=set_to_none)