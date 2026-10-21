# Attention sinks and Schatten-4 initialization: Colab ablation

## Objective and design

Determine whether Schatten-4 normalization inside Muon's Newton–Schulz iteration
reduces attention sinks, versus intervening at attention itself. Six paired
arms: AdamW, Muon with Frobenius scaling, Muon with Schatten-4 scaling, Muon/S4
plus a column-concentration penalty, Muon/S4 plus learned null attention, and
Muon/S4 with both changes.

This is a small **mechanistic testbench**, not a pretrained-LLM result.
All arms use the same seed and paired training examples.

## Launch and capture

An authenticated Colab CLI and T4 GPU entitlement are required. From repo root:

    colab run --gpu T4 --keep -s gcl-s4-sink-full --timeout 3600 \
      experiments/attention_sink_s4.py \
      --steps 260 --batch 32 --span 20 --width 96 --layers 2 --heads 4 \
      --eval-batches 4 --log-every 20 --lambda-conc 10 --wandb-mode offline

    colab download -s gcl-s4-sink-full \
      /content/gcl_s4_sink_results/results.zip attention-sink-results.zip
    colab stop -s gcl-s4-sink-full

CPU diagnostic smoke:

    python experiments/attention_sink_s4.py --steps 2 --batch 4 --span 8 \
      --width 32 --layers 1 --arms adamw muon_s4_null_conc \
      --eval-batches 1 --self-test --wandb-mode disabled --out /tmp/s4sink

Outputs: config.json, positive_control.json, metrics.jsonl,
summary.json, summary.csv, results.zip. W&B is offline by default and
its records are included in the bundle; online mode is opt-in.

## Evaluation

The task is causal next-token modeling on BOS + random 20-token span +
separator + repeated span + EOS. The task loss combines whole-sequence CE
and repeated-span CE; held-out CE and copy accuracy measure utility.

- BOS mass measures mean first-key probability on late queries.
- BOS excess subtracts the causal mask-aware uniform real-token baseline.
  Negative values mean no BOS sink relative to that baseline.
- Max column mass measures concentrated attention across late queries.
- Fourth-order penalty measures excess column energy above mask-aware
  uniform allocation; it is an S4-inspired statistic, NOT the full
  Schatten-4 norm of the attention matrix.
- Null mass measures probability assigned to the zero-valued null route.

The optimizer uses the S4 norm as one *global* Newton–Schulz scaling
denominator, not singular-value-wise power-four shrinkage. The standalone
Muon optimizer is an educational reference rather than the production
MuonSphere optimizer. Positive control verifies a 16x16 single-key
attention sink has S4 norm 4, versus 1 for uniform attention.

## T4 run: 2026-10-21

PyTorch 2.11.0+cu130; seed 20261021; 260 steps/arm; batch 32.
Full machine-readable results are committed alongside this document.

| Arm | CE | Copy | BOS mass | BOS excess | Max column | 4th penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 1.6586 | 1.000 | .0201 | -.0124 | .0686 | 1.0573 |
| Muon Frobenius | 1.6561 | 1.000 | .0143 | -.0182 | .0954 | 1.0908 |
| Muon S4 | 1.6561 | 1.000 | .0148 | -.0177 | .1098 | 1.1693 |
| Muon S4 + concentration | 1.6615 | .998 | .0367 | +.0043 | .0557 | .0030 |
| Muon S4 + null | 1.6563 | 1.000 | .0098 | -.0226 | .1402 | 1.7973 |
| Muon S4 + both | 1.6605 | .998 | .0318 | +.0065 | .0455 | .0035 |

S4 normalization alone does not reduce BOS sink mass compared with the
Frobenius option in this test. Fourth-order attention regularization
dramatically reduces its targeted measure; however it also raises BOS
mass and slightly raises validation CE. The null route reduces BOS mass
in its standalone arm but does not reduce overall column concentration.

None of the unregularized arms exhibits positive BOS excess at
convergence. Therefore the experiment does not establish suppression of
pathological *natural* sinks. It establishes that the harness works and
that attention concentration responds to the direct regularizer.

## Limitations and next experiment

One seed; optimizer and step budgets not hyperparameter- or FLOP-matched.
Copying is nearly solved by all methods and does not reproduce LLM sinks.
A stronger test is a controlled BOS-logit bias stressor, multi-seed runs,
fixed-loss controls, then natural-language LM experiments at much larger
scale with long-context evaluation. No claim of sink mitigation in real
LLMs follows from these results.
