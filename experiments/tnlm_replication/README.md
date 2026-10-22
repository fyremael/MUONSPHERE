# Multi-seed Force-sink S4 replication and downstream retrieval

This experiment replicates the published A100 structural sink-induction
comparison under three independently initialized GPT-2 Small models
(seeds 101, 202, 303). Each seed has exactly three AdamW arms:
`fsm_tnlm`, `fsm_tnlm_s4`, and `fsm_tnlm_s4_col`. Both S4 and
the extra targeted column penalty use coefficient 0.15.
The model is 124,439,808 parameters, randomly initialized.
Natural language is FineWeb-Edu `sample-10BT`; there is no synthetic
attention logit bias or pretrained checkpoint.

Training settings match the existing 2026-10-21 positive control:
context length 512, microbatch 2, gradient accumulation 4 (effective
batch 8), 4,000 steps per run, 16,384,000 processed token positions
per run; AdamW peak LR 3e-4, 100 warmup steps, cosine decay,
betas (0.9,0.95), weight decay 0.1, BF16 forward, TF32 enabled,
fused AdamW, zero dropout; structural Force-sink Masking reserves
16 dimensions in a position-zero path. Validation includes held-out
next-token CE at 512 and 1024 tokens. Training samples are
deterministically paired for each seed and arm. The separate
training runs are truly independent between seeds. The corpus has
60 million training and 4 million validation tokens, split by
document ID SHA-256. All allocated runtimes regenerate the exact
previous corpus token-file hashes, verified before launch.

## Downstream zero-shot retrieval

At the end of training, with no gradient updates:

- **Verbatim passage retrieval:** Eight tokens from a disjoint
  validation passage appear near the start of a context, and the
  same eight-token target is scored near the end. We compare intact
  earlier evidence against an *otherwise identical* input with
  the early evidence replaced by a different passage. The primary
  context-use diagnostic is held-out target cross-entropy reduction
  in nats, `copy_lift = CE_corrupted - CE_intact`. Positive values
  indicate benefit from available early evidence. Direct
  repeated-span top-1 token accuracy is also measured.
- **Four-choice passkey retrieval:** A passkey is presented near
  the start of a natural-text-filled context and queried near the
  end. The correct next-token score is ranked against three fixed
  alternatives; random-choice accuracy is 25%. Replacing the
  earlier passkey provides an ablation test that can detect
  prompt-insensitive scores.

Both tests use 512 and 1024 token contexts, 24 deterministic examples
at each length, identical across all models and seeds. They are
**zero-shot probes**, not supervised retrieval training or a claim that
16 million processed pretraining tokens suffices for mature instruction
following. Interpret retrieval gains only if intact-versus-ablated
conditions show material information benefit rather than chance-level
forced choices. The tests also report per-seed deviations.

## Hardware utilization

Two parallel **A100-SXM4-40GB** runtimes are allocated; a third was
denied by the allocator. Seed 202 and seed 101 are run as separate
processes on the first A100, with isolated outputs; seed 303 runs on
the second. Initial measured A100 GPU usage increased from roughly
54% at 6.3 GiB allocated (one training process) to **99% at 11.0 GiB**
(two processes), without changes to each arm's optimizer or effective
batch. This saturates the available GPU compute using only ~27% of
its VRAM. Each complete run records peak GPU memory and
tokens-per-second. All GPU sessions must be released after durable
receipt recovery.

## Execution and adjudication

- `prepare_fineweb.py`: reconstruct exact token-file hashes
- `train_replicate.py`: full 3-arm, one-seed trainer with W&B offline
- `retrieval_eval.py`: deterministic, paired held-out context-use probes
- `launch_101.py`, `launch_202.py`, `launch_303.py`:
  isolated seed launch, verified corpus provenance
- `watch_seed.sh`: periodic per-arm/seed metrics mirroring
- `analyze.py`: refuses to emit COMPLETE unless all 9 arms
  have exactly 4,000 steps, exact dataset hashes, matched
  initialization within seed, and two retrieval context results;
  computes paired three-seed mean, sample SD, and t-based 95% intervals

The published seed-101 baseline is an independent reproducibility
reference but does not substitute for the new seed-101 retrieval
result. Failed or disconnected jobs must not be counted as complete.

