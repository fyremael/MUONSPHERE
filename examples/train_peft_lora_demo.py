"""Minimal PEFT LoRA spectral-constraint smoke demo."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from muonsphere.peft import PeftSpectralConstraintManager, inject_lora


class Toy(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.mlp.fc2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return self.attn.proj(self.attn.qkv(x)[..., :64]) + self.mlp.fc2(F.gelu(self.mlp.fc1(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--retract_every", type=int, default=10)
    args = p.parse_args()

    model = inject_lora(Toy(), r=4, alpha=8.0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    constraints = PeftSpectralConstraintManager(c_radius=1.0, retract_every=args.retract_every)

    for step in range(1, args.steps + 1):
        x = torch.randn(8, 16, 64)
        y = torch.randn_like(x)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        stats = constraints.retract(model)
        if step == 1 or step % 10 == 0:
            print(step, float(loss), stats.as_dict())


if __name__ == "__main__":
    main()
