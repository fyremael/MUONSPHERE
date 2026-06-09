"""PEFT hooks for MUONSPHERE.

This module implements the pragmatic PEFT path: constrain the effective LoRA
operator `W_eff = W0 + s ΔW` by adjusting the scalar delta scale `s`.  This keeps
standard PEFT training lightweight while preserving the MODULUS principle that
operator gain should be explicitly instrumented and governed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x) + x.new_tensor(eps))


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / _norm(x, eps)


class LoRALinear(nn.Module):
    """Drop-in `nn.Linear` replacement with LoRA and a controllable delta scale."""

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / max(1, r)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.register_buffer("delta_scale", torch.tensor(1.0), persistent=True)
        self.reset_parameters()

    @classmethod
    def from_linear(cls, lin: nn.Linear, r: int = 8, alpha: float = 16.0, freeze_base: bool = True) -> "LoRALinear":
        mod = cls(lin.in_features, lin.out_features, r=r, alpha=alpha, bias=lin.bias is not None).to(lin.weight.device)
        mod.weight.data.copy_(lin.weight.data)
        if lin.bias is not None and mod.bias is not None:
            mod.bias.data.copy_(lin.bias.data)
        if freeze_base:
            mod.weight.requires_grad_(False)
            if mod.bias is not None:
                mod.bias.requires_grad_(False)
        return mod

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_delta_scale(self) -> float:
        return float(self.delta_scale.item())

    @torch.no_grad()
    def set_delta_scale(self, s: float) -> None:
        self.delta_scale.fill_(float(s))

    def delta_matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.scaling * (self.lora_B @ (self.lora_A @ v))

    def delta_rmatvec(self, u: torch.Tensor) -> torch.Tensor:
        return self.scaling * (self.lora_A.t() @ (self.lora_B.t() @ u))

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.weight @ v + self.get_delta_scale() * self.delta_matvec(v)

    def rmatvec(self, u: torch.Tensor) -> torch.Tensor:
        return self.weight.t() @ u + self.get_delta_scale() * self.delta_rmatvec(u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.get_delta_scale() * self.scaling * lora


@dataclass
class PeftStats:
    n_modules: int = 0
    adjusted: int = 0
    sigma_over_radius: float = float("nan")
    delta_scale_mean: float = float("nan")

    def as_dict(self, prefix: str = "peft/") -> Dict[str, float]:
        return {
            prefix + "n_modules": float(self.n_modules),
            prefix + "adjusted": float(self.adjusted),
            prefix + "sigma_over_radius": float(self.sigma_over_radius),
            prefix + "delta_scale_mean": float(self.delta_scale_mean),
        }


class _Power:
    def __init__(self, niter: int = 3, eps: float = 1e-8):
        self.niter = niter
        self.eps = eps
        self.cache: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def sigma(self, op: LoRALinear) -> float:
        key = id(op)
        v = self.cache.get(key)
        if v is None or v.numel() != op.in_features or v.device != op.weight.device:
            v = _unit(torch.randn(op.in_features, device=op.weight.device, dtype=op.weight.dtype), self.eps)
        for _ in range(self.niter):
            u = _unit(op.matvec(v), self.eps)
            v = _unit(op.rmatvec(u), self.eps)
        self.cache[key] = v
        return float(_norm(op.matvec(v), self.eps).item())


class PeftSpectralConstraintManager:
    """Constrain LoRA effective operators by reducing each module's delta scale."""

    def __init__(self, c_radius: float = 1.0, power_niter: int = 3, retract_every: int = 10, max_bisect: int = 14):
        self.c_radius = c_radius
        self.power = _Power(power_niter)
        self.retract_every = retract_every
        self.max_bisect = max_bisect
        self.step_count = 0

    def modules(self, model: nn.Module) -> List[LoRALinear]:
        return [m for m in model.modules() if isinstance(m, LoRALinear)]

    @staticmethod
    def radius(m: LoRALinear, c: float) -> float:
        return c * math.sqrt(m.out_features / m.in_features)

    @torch.no_grad()
    def retract(self, model: nn.Module, force: bool = False) -> PeftStats:
        self.step_count += 1
        mods = self.modules(model)
        if not mods:
            return PeftStats()
        if not force and self.retract_every > 1 and self.step_count % self.retract_every:
            scales = torch.tensor([m.get_delta_scale() for m in mods])
            return PeftStats(len(mods), 0, float("nan"), float(scales.mean().item()))
        ratios, scales = [], []
        adjusted = 0
        for m in mods:
            R = self.radius(m, self.c_radius)
            s0 = m.get_delta_scale()
            sigma0 = self.power.sigma(m)
            ratios.append(sigma0 / (R + 1e-12))
            if sigma0 <= R:
                scales.append(s0)
                continue
            lo, hi = 0.0, s0
            m.set_delta_scale(0.0)
            if self.power.sigma(m) > R:
                scales.append(0.0)
                adjusted += 1
                continue
            for _ in range(self.max_bisect):
                mid = 0.5 * (lo + hi)
                m.set_delta_scale(mid)
                if self.power.sigma(m) > R:
                    hi = mid
                else:
                    lo = mid
            m.set_delta_scale(lo)
            scales.append(lo)
            adjusted += 1
        return PeftStats(len(mods), adjusted, float(torch.tensor(ratios).mean().item()), float(torch.tensor(scales).mean().item()))


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, target_substrings: Optional[List[str]] = None) -> nn.Module:
    """Replace selected `nn.Linear` modules with `LoRALinear`."""
    targets = target_substrings or ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
    named = dict(model.named_modules())
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not any(t in name for t in targets) or "lm_head" in name:
            continue
        parent_name, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if parent_name == "" else named[parent_name]
        setattr(parent, child, LoRALinear.from_linear(mod, r=r, alpha=alpha, freeze_base=True))
    return model
