import torch
import torch.nn as nn

from muonsphere.atom import registry
from muonsphere.optim import MuonSphere
from muonsphere.peft import LoRALinear, PeftSpectralConstraintManager, inject_lora


class Block(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.dim = dim
        self.attn.n_heads = heads
        self.attn.head_dim = dim // heads
        self.attn.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, 2 * dim, bias=False)
        self.mlp.fc2 = nn.Linear(2 * dim, dim, bias=False)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block()])


def test_atom_builder_and_muonsphere_step():
    model = Toy()
    atoms = registry.build(model, "tiny_transformer", atomic_qkv_per_head=True)
    assert atoms
    opt = MuonSphere(atoms, lr=1e-3)
    loss = sum((p ** 2).sum() for p in model.parameters())
    loss.backward()
    stats = opt.step()
    assert stats.n_atoms > 0


def test_lora_matvec_and_constraint():
    lin = nn.Linear(16, 12, bias=False)
    lora = LoRALinear.from_linear(lin, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.02)
        v = torch.randn(16)
        explicit = lora.weight + lora.get_delta_scale() * lora.scaling * (lora.lora_B @ lora.lora_A)
        assert torch.allclose(lora.matvec(v), explicit @ v, atol=1e-5)
    model = nn.Sequential(lora)
    mgr = PeftSpectralConstraintManager(c_radius=0.25, retract_every=1)
    before = lora.get_delta_scale()
    stats = mgr.retract(model, force=True)
    assert stats.n_modules == 1
    assert lora.get_delta_scale() <= before + 1e-9


def test_inject_lora():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.attn.qkv = nn.Linear(8, 24, bias=False)
    model = inject_lora(M(), r=2, alpha=4.0, target_substrings=["attn.qkv"])
    assert isinstance(model.attn.qkv, LoRALinear)
