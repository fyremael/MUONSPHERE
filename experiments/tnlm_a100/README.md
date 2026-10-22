# TNLM and Force-sink induction: A100 natural-language testbench

Objective: test whether a pronounced positional attention sink can be
*learned* in random-initialization GPT-2 on FineWeb-Edu by removing the
first prediction's loss, without adding an attention logit bias. Then
compare S4 attention regularization against the same induced condition.

Motivation: Peng et al., arXiv:2603.06591 (2026); Gu et al., ICLR 2025,
arXiv:2410.10781. These report dependence of position-zero sink formation
on causal structure, optimization and the language-model training objective.

All conditions use GPT-2 Small 124M from random initialization; AdamW;
GPT-2 BPE; FineWeb-Edu sample-10BT with a fixed document-disjoint split,
60M train and 4M validation tokens; fixed EOT at position zero, 512-token
training context, 1024-token validation diagnostic, no artificial
attention-logit injection.

Experimental arms:
- lm: ordinary causal next-token prediction. Original completed 4,000-step
  A100 baseline remains available from earlier tranche.
- tnlm: ignores only the prediction at sequence position zero. Hugging Face
  GPT2 internally shifts labels; therefore the required operation is
  labels[:,1]=-100, not labels[:,0]=-100 (already ignored).
- fsm_tnlm: TNLM plus fixed nonlearned P0 channel mask before pre-LN1
  and pre-LN2 in every block. P0 is limited to k=16 reserved dimensions;
  subsequent positions use the other 752 dimensions. This is a structural
  positive control, distinct from wholly spontaneous sink formation.
- tnlm_s4: TNLM plus true Schatten-4^4 on 128 sampled query rows of
  attention deviation from causal-uniform, with all key columns retained.
- tnlm_s4_col: previous S4 penalty plus fourth-order positive-excess
  attention column concentration penalty.
- fsm_tnlm_s4 and fsm_tnlm_s4_col: matching regularized arms when
  only structural force-sink induction produces a qualified sink.

All seed-matched arms see identical training windows and initial weight
sample fingerprints. AdamW betas=(0.9,0.95), weight_decay=0.1; peak LR 3e-4,
100 warmup steps, cosine schedule, BF16 model forward and FP32 S4 penalty.

Initial 4,000-step induction pilot: seed 101, arms tnlm and fsm_tnlm.
Track per-layer and global BOS probability, max key-column concentration,
fraction of heads >30% BOS attention, held-out full autoregressive
cross-entropy, attention entropy and context-length generalization.

Proposed pronounced-sink admission gate: in at least one layer, >=25%
of heads attend BOS with mean probability >30%, in three consecutive
validation checkpoints. Later confirm across at least three seeds.
The unregularized induction must pass this gate before claiming that S4
prevents the induced sink. A lower BOS metric alone, with degraded loss,
relocated sink or mere attention diffusion, is not a mitigation success.

Verification: test_mask.py confirms first-query logit gradient is zero
under TNLM, later-query gradients remain, and structural mask partitions
the exact intended channels. Both tests passed on Colab A100.

Commands:
  colab new --gpu A100 -s gcl-tnlm-sink-a100
  colab upload -s gcl-tnlm-sink-a100 train_tnlm.py /content/gcl_tnlm_train.py
  colab exec -s gcl-tnlm-sink-a100 -f launch_induction.py
  bash watch_induction.sh

Training and W&B telemetry are stored in the Colab run folder. External
evidence mirroring saves metrics.jsonl, summary.json, config.json and
initial.jsonl. The GPU must be explicitly released after evidence capture.

## One-seed calibration results (2026-10-21)

The 4,000-step TNLM-only arm did not reach the prespecified sustained
sink gate; TNLM plus Force-sink Masking did. The qualifying baseline
was verified with three successive checkpoints in the same layer.
The matched S4-only and S4-plus-column regularized arms were also
completed. The first S4-only controller failed during unnecessary
corpus preparation; its failed receipt is retained while a direct
replay ran to completion on verified cached tokens. Results and
quantitative limitations are in
`docs/attention_sink_tnlm_fsm_20261021_report.md`. This is a single
seed and therefore not a replicated finding.

Custody note: the completed FSM baseline replay and S4-only direct replay
have private offline W&B archives recovered on CHAD. Recovery of the
S4-plus-column arm's offline W&B binary was not verified; its complete
public JSONL metrics, config, and summary were captured and admitted.
