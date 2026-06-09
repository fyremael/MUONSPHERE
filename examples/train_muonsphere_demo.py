"""Minimal MuonSphere smoke demo."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from muonsphere.atom import registry
from muonsphere.optim import MuonSphere


class Block(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.dim = dim
        self.attn.n_heads = heads
        self.attn.head_dim = dim // heads
        self.attn.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.mlp.fc2 = nn.Linear(4 * dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm(x))))


class ToyModel(nn.Module):
    def __init__(self, dim=64, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim=dim) for _ in range(layers)])
        self.head = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--atomic_qkv_per_head", action="store_true")
    args = p.parse_args()

    model = ToyModel()
    atoms = registry.build(model, "tiny_transformer", atomic_qkv_per_head=args.atomic_qkv_per_head)
    muon = MuonSphere(atoms, lr=1e-3)
    adam = torch.optim.AdamW([p for p in model.parameters() if p not in {a.param for a in atoms}], lr=1e-3)

    for step in range(1, args.steps + 1):
        x = torch.randn(8, 16, 64)
        y = torch.randn_like(x)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        stats = muon.step()
        adam.step()
        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            print(step, float(loss), stats.as_dict())


if __name__ == "__main__":
    main()
