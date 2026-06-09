"""MuonSphere optimizer.

MuonSphere constrains selected 2D operators to a spectral sphere
`||W||_2 = R`, with `R = c * sqrt(d_out / d_in)`.  The update direction is a
Muon-style polar-factor approximation to a matrix sign direction.  This gives a
practical first implementation of MODULUS-style operator control: state vectors
may be normalized by the architecture, while operators are kept gain-controlled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .atom import Atomic2DView


def _norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x) + x.new_tensor(eps))


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / _norm(x, eps)


@dataclass
class SpectralTriplet:
    u: torch.Tensor
    v: torch.Tensor
    sigma: torch.Tensor


class SpectralPowerIter:
    """Cached power iteration for top singular triplets."""

    def __init__(self, niter: int = 2, eps: float = 1e-8):
        self.niter = niter
        self.eps = eps
        self.cache: Dict[object, torch.Tensor] = {}

    @torch.no_grad()
    def estimate(self, W: torch.Tensor, key: object) -> SpectralTriplet:
        _, n = W.shape
        v = self.cache.get(key)
        if v is None or v.numel() != n or v.device != W.device or v.dtype != W.dtype:
            v = _unit(torch.randn(n, device=W.device, dtype=W.dtype), self.eps)
        for _ in range(self.niter):
            u = _unit(W @ v, self.eps)
            v = _unit(W.t() @ u, self.eps)
        u = W @ v
        sigma = _norm(u, self.eps)
        u = u / (sigma + W.new_tensor(self.eps))
        self.cache[key] = v
        return SpectralTriplet(u, v, sigma)


def msign_polar(G: torch.Tensor, niter: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """Approximate the polar factor of `G` with Newton--Schulz iterations."""
    m, n = G.shape
    X = G / (torch.linalg.norm(G, ord="fro") + G.new_tensor(eps))
    if m >= n:
        I = torch.eye(n, device=G.device, dtype=G.dtype)
        for _ in range(niter):
            X = 0.5 * X @ (3.0 * I - X.t() @ X)
    else:
        I = torch.eye(m, device=G.device, dtype=G.dtype)
        for _ in range(niter):
            X = 0.5 * (3.0 * I - X @ X.t()) @ X
    return X


@dataclass
class MuonSphereStats:
    n_atoms: int = 0
    sigma_over_radius_before: float = float("nan")
    sigma_over_radius_after: float = float("nan")
    tangency: float = float("nan")

    def as_dict(self, prefix: str = "muon/") -> dict[str, float]:
        return {
            prefix + "n_atoms": float(self.n_atoms),
            prefix + "sigma_over_radius_before": float(self.sigma_over_radius_before),
            prefix + "sigma_over_radius_after": float(self.sigma_over_radius_after),
            prefix + "tangency": float(self.tangency),
        }


class MuonSphere(torch.optim.Optimizer):
    """Spectral-sphere Muon-style optimizer over `Atomic2DView` objects."""

    def __init__(
        self,
        atoms: Sequence[Atomic2DView],
        lr: float = 2e-3,
        c_radius: float = 1.0,
        use_mup_lr: bool = True,
        tangent_mode: bool = False,
        power_niter: int = 2,
        msign_niter: int = 5,
        max_grad_norm: Optional[float] = 1.0,
        eps: float = 1e-8,
    ):
        params = list({a.param for a in atoms})
        super().__init__([{"params": params}], dict(lr=lr))
        self.atoms = list(atoms)
        self.c_radius = c_radius
        self.use_mup_lr = use_mup_lr
        self.tangent_mode = tangent_mode
        self.msign_niter = msign_niter
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.power = SpectralPowerIter(power_niter, eps)

    @staticmethod
    def radius(m: int, n: int, c: float) -> float:
        return c * math.sqrt(m / n)

    @torch.no_grad()
    def _retract(self, W: torch.Tensor, R: float, trip: SpectralTriplet) -> None:
        sigma = float(trip.sigma.item())
        if math.isfinite(sigma) and sigma > self.eps:
            W.mul_(R / sigma)

    @torch.no_grad()
    def _clip(self) -> None:
        if self.max_grad_norm is None:
            return
        grads = [a.grad_view() for a in self.atoms if a.grad_view() is not None]
        if not grads:
            return
        total = math.sqrt(sum(float((g * g).sum().item()) for g in grads) + self.eps)
        if total > self.max_grad_norm:
            scale = self.max_grad_norm / (total + self.eps)
            for g in grads:
                g.mul_(scale)

    @torch.no_grad()
    def _solve_lambda(self, G: torch.Tensor, Theta: torch.Tensor) -> float:
        def h(lam: float) -> float:
            return float((Theta * msign_polar(G + lam * Theta, self.msign_niter, self.eps)).sum().item())

        h0 = h(0.0)
        if abs(h0) < 1e-6:
            return 0.0
        direction = -1.0 if h0 > 0 else 1.0
        lo, hi = 0.0, direction
        hlo, hhi = h0, h(hi)
        for _ in range(12):
            if (hlo > 0 > hhi) or (hlo < 0 < hhi):
                break
            hi *= 2.0
            hhi = h(hi)
        else:
            return 0.0
        a, b = sorted([lo, hi])
        fa = h(a)
        for _ in range(20):
            mid = 0.5 * (a + b)
            fm = h(mid)
            if abs(fm) < 1e-6:
                return mid
            if (fa > 0 > fm) or (fa < 0 < fm):
                b = mid
            else:
                a, fa = mid, fm
        return 0.5 * (a + b)

    @torch.no_grad()
    def step(self, closure=None) -> MuonSphereStats:  # type: ignore[override]
        if closure is not None:
            with torch.enable_grad():
                closure()
        lr = self.param_groups[0]["lr"]
        self._clip()
        rb, ra, tang = [], [], []
        for atom in self.atoms:
            G = atom.grad_view()
            if G is None:
                continue
            W = atom.view()
            m, n = W.shape
            R = self.radius(m, n, self.c_radius)
            lr_eff = lr * (math.sqrt(m / n) if self.use_mup_lr else 1.0)
            trip = self.power.estimate(W, atom.key())
            rb.append(float(trip.sigma.item()) / (R + 1e-12))
            self._retract(W, R, trip)
            Theta = torch.outer(trip.u, trip.v)
            if self.tangent_mode:
                lam = self._solve_lambda(G, Theta)
                Phi = msign_polar(G + lam * Theta, self.msign_niter, self.eps)
            else:
                Phi = msign_polar(G, self.msign_niter, self.eps)
            tang.append(float((Theta * Phi).sum().item()))
            W.add_(Phi, alpha=-lr_eff)
            trip2 = self.power.estimate(W, atom.key())
            ra.append(float(trip2.sigma.item()) / (R + 1e-12))
            self._retract(W, R, trip2)
        def mean(xs):
            return float(torch.tensor(xs).mean().item()) if xs else float("nan")
        return MuonSphereStats(len(rb), mean(rb), mean(ra), mean(tang))
