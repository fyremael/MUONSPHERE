# Run Matrix

## Full-weight path

| Run | Optimizer | Atomization | Tangent mode | Purpose |
|---|---|---|---|---|
| A0 | AdamW | none | no | baseline |
| A1 | MuonSphere | whole matrix | no | retraction baseline |
| A2 | MuonSphere | per-head QKV | no | semantic atomization |
| A3 | MuonSphere | per-head QKV | yes | tangent solve stress test |

## PEFT path

| Run | Adapter | Constraint | Frequency | Purpose |
|---|---|---|---|---|
| P0 | LoRA | none | n/a | baseline |
| P1 | LoRA | monitor | 10 | observe sigma/R |
| P2 | LoRA | delta-scale retraction | 10 | practical stability path |
| P3 | LoRA | delta-scale retraction | 1 | strict stress test |

## Standard knobs

- c_radius: 0.5, 1.0, 1.5
- power_niter: 1, 2, 3, 5
- msign_niter: 3, 5, 8
- atomic_qkv_per_head: false, true
- LoRA rank: 4, 8, 16, 32
- PEFT retract_every: 1, 10, 50, 100
