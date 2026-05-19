# MUONSPHERE

**MuonSphere** is a practitioner-oriented research scaffold for nGPT/MODULUS-style spectral operator control.

It contains two complementary paths:

1. **Full-weight nGPT/MODULUS path** — `MuonSphere`, a spectral-sphere constrained Muon-style optimizer for selected 2D transformer operators, with an `AtomBuilder` registry for architecture-specific atomization.
2. **PEFT path** — LoRA wrappers plus an effective-operator spectral constraint manager for `W_eff = W0 + s ΔW`.

The organizing idea is simple: if normalized transformer streams control the *state*, MuonSphere controls the *operators* that move that state. For each atomic matrix, the target invariant is

```text
||W||_2 ≈ R,        R = c * sqrt(d_out / d_in)
```

For LoRA/PEFT, the monitored operator is the effective matrix:

```text
W_eff = W0 + s ΔW
```

## Install

```bash
pip install -e .
```

## Smoke tests

```bash
python -m ngpt_modulus.muonsphere_atomregistry --run_tests
python -m ngpt_modulus.peft_lora_constraints --run_tests
```

Or:

```bash
pytest -q
```

## Demos

```bash
python examples/train_muonsphere_demo.py --steps 50 --atomic_qkv_per_head
python examples/train_peft_lora_demo.py --steps 50 --retract_every 10
```

## Repository layout

```text
src/ngpt_modulus/muonsphere_atomregistry.py     # MuonSphere + AtomBuilder registry
src/ngpt_modulus/peft_lora_constraints.py      # LoRA + PEFT spectral constraints
examples/                                      # runnable demos
tests/                                         # CPU smoke tests
configs/                                       # starter configs
docs/                                         # spec, run matrix, metrics, integration guide
```

## Status

This repository is a working research/practitioner scaffold. It is meant to make experiments concrete, auditable, and easy to extend. The next hardening passes should add distributed-training integration, richer telemetry, and real-model ablations.
