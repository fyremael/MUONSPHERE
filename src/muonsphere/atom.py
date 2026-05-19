"""Atom builders for MUONSPHERE.

The optimizer should not know a model's architecture.  This module defines the
small abstraction that maps a model to atomic two-dimensional operators.  Each
atom is a live view into a parameter tensor.  The same idea later extends to
PEFT effective weights, where an atom may represent W_eff = W0 + ΔW rather than a
raw parameter slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Atomic2DView:
    """Live view into a 2D parameter, optionally restricted to rows/columns."""

    param: torch.nn.Parameter
    row_slice: Optional[Tuple[int, int]] = None
    col_slice: Optional[Tuple[int, int]] = None
    name: str = ""

    def key(self):
        return (id(self.param), self.row_slice, self.col_slice)

    def view(self) -> torch.Tensor:
        rs = slice(None) if self.row_slice is None else slice(*self.row_slice)
        cs = slice(None) if self.col_slice is None else slice(*self.col_slice)
        return self.param[rs, cs]

    def grad_view(self) -> Optional[torch.Tensor]:
        if self.param.grad is None:
            return None
        rs = slice(None) if self.row_slice is None else slice(*self.row_slice)
        cs = slice(None) if self.col_slice is None else slice(*self.col_slice)
        return self.param.grad[rs, cs]


class AtomBuilderRegistry:
    """Registry of architecture-specific atom builders.

    A builder is a callable `(model, **kwargs) -> list[Atomic2DView]`.
    Register one builder per model family so MuonSphere stays architecture-free.
    """

    def __init__(self):
        self._builders: Dict[str, Callable[..., List[Atomic2DView]]] = {}

    def register(self, name: str, fn: Callable[..., List[Atomic2DView]]) -> None:
        self._builders[name] = fn

    def names(self) -> List[str]:
        return sorted(self._builders)

    def build(self, model: nn.Module, name: str = "generic", **kwargs) -> List[Atomic2DView]:
        if name == "auto":
            name = "tiny_transformer" if hasattr(model, "blocks") else "generic"
        if name not in self._builders:
            raise KeyError(f"Unknown atom builder {name!r}; available={self.names()}")
        return _dedupe(self._builders[name](model, **kwargs))


def _dedupe(atoms: List[Atomic2DView]) -> List[Atomic2DView]:
    seen, out = set(), []
    for atom in atoms:
        if atom.key() in seen:
            continue
        seen.add(atom.key())
        out.append(atom)
    return out


def generic_builder(model: nn.Module, **_) -> List[Atomic2DView]:
    """Fallback: all trainable 2D weights except obvious embeddings/output heads."""
    atoms: List[Atomic2DView] = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.ndim == 2 and not any(s in name for s in ("emb", "lm_head", "output")):
            atoms.append(Atomic2DView(p, name=name))
    return atoms


def tiny_transformer_builder(model: nn.Module, atomic_qkv_per_head: bool = True, **_) -> List[Atomic2DView]:
    """Builder for the demo GPT-style architecture.

    QKV can be split per head so the spectral constraint acts at the same
    granularity as attention computation.
    """
    atoms: List[Atomic2DView] = []
    for li, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "qkv"):
            qkv = attn.qkv
            if atomic_qkv_per_head and all(hasattr(attn, x) for x in ("n_heads", "head_dim", "dim")):
                dim, head_dim = int(attn.dim), int(attn.head_dim)
                for label, base in (("q", 0), ("k", 1), ("v", 2)):
                    for h in range(int(attn.n_heads)):
                        r0 = base * dim + h * head_dim
                        atoms.append(Atomic2DView(qkv.weight, (r0, r0 + head_dim), None, f"blocks.{li}.attn.{label}.{h}"))
            else:
                atoms.append(Atomic2DView(qkv.weight, name=f"blocks.{li}.attn.qkv"))
            atoms.append(Atomic2DView(attn.proj.weight, name=f"blocks.{li}.attn.proj"))
        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            atoms.append(Atomic2DView(mlp.fc1.weight, name=f"blocks.{li}.mlp.fc1"))
            atoms.append(Atomic2DView(mlp.fc2.weight, name=f"blocks.{li}.mlp.fc2"))
    return atoms


registry = AtomBuilderRegistry()
registry.register("generic", generic_builder)
registry.register("tiny_transformer", tiny_transformer_builder)
