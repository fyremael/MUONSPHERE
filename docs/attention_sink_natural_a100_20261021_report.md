# A100 natural-language attention-sink continuation experiment

**Status: completed; 15/15 training arms, 3/3 seeds, 5/5 methods.**
Execution: 2026-10-21 UTC. GPU: NVIDIA A100-SXM4-40GB, released
after retrieving all run artifacts.

## What was actually tested

A real **GPT-2 Small 124,439,808-parameter pretrained model** was
continued on text from FineWeb-Edu (`HuggingFaceFW/fineweb-edu`,
`sample-10BT`). GPT-2 BPE with 50,257 vocabulary entries. Dataset
materialized to 12,000,000 train and 1,000,000 held-out tokens;
document-level split by SHA-256 ID hash. This was **continued
pretraining**, not a random-initialization pretraining experiment.
No attention-logit bias or other artificial sink inducer was used.
The EOT token (GPT-2 token 50256) was prepended as a standard start
marker to each sampled language-model sequence. All arms used the
same original pretrained weights and the same validation samples.

Model: 12 layers, 768 hidden dimensions, 12 attention heads,
native learned positional embeddings. Five arms with paired RNG
seeds (101, 202, 303), 320 optimization steps per arm, microbatch 2,
gradient accumulation 2, context 512, 655,360 token positions per
run. Total positions processed: 9,830,400 over all 15 arms, with
repeated draws from the fixed train corpus. The vocabulary and
architecture are ordinary GPT-2; the first-token sink is not
artificially injected. The hold-out includes 512-token and separate
1,024-token evaluations.

Five comparisons:
- Muon with Frobenius Newton–Schulz initialization.
- Muon with Schatten-4 Newton–Schulz initialization.
- Muon/S4 plus differentiable fourth-order excess attention-column penalty.
- Standard AdamW continued pretraining.
- AdamW plus the same fourth-order attention penalty.

Regularization coefficient 0.03, Muon learning rate 0.001, AdamW
learning rate 3e-5, no dropout; BF16 A100 forward,
float32 polar normalization. All use next-token cross-entropy only,
plus the named regularizer. This is an **S4-inspired attention-column
penalty**, not the full Schatten-4 norm of the attention matrix.

## Initial natural sink verification

The pretrained model (before continuation) on the 512-token validation
window displayed BOS real-token attention share ~37.8%, BOS excess
~37.4 percentage points over causal uniform, and about two-thirds of
layer/head units gave the BOS key >30% of later-query attention.
All fifteen initial checkpoint evaluations passed the specified
natural sink gate. No injected attention bias was present.

## Measured results (three seeds)

Values are **means ± sample standard deviations**; BOS and max-column
are fractions of total real-token attention.

| Method | BOS share | Maximum column share | Held-out CE, 512 | Held-out CE, 1024 |
| --- | ---: | ---: | ---: | ---: |
| Muon/Frobenius | 0.3768 ± 0.0013 | 0.3781 ± 0.0012 | 3.2572 ± 0.0140 | 3.2074 ± 0.0074 |
| Muon/S4 | 0.3754 ± 0.0016 | 0.3766 ± 0.0016 | 3.2570 ± 0.0148 | 3.2057 ± 0.0085 |
| Muon/S4 + column penalty | **0.0902 ± 0.0021** | **0.0918 ± 0.0022** | 3.2960 ± 0.0097 | 3.2857 ± 0.0059 |
| AdamW | 0.3749 ± 0.0016 | 0.3761 ± 0.0015 | **3.2451 ± 0.0080** | **3.1846 ± 0.0072** |
| AdamW + column penalty | 0.0983 ± 0.0014 | 0.0996 ± 0.0014 | 3.2933 ± 0.0086 | 3.2675 ± 0.0105 |

On 1,024-token windows, mean BOS attention shares were
Muon/S4: **0.3464** and Muon/S4 plus penalty: **0.0718**.
All seeded runs showed reduction of both BOS concentration
and maximum-column concentration under the direct penalty.

### Results and costs

**Muon's Schatten-4 initialization alone is not a sink-mitigation
mechanism.** Its mean BOS share was 37.54%, versus 37.68% with
Frobenius initialization. The 512-token CE difference was negligible
(3.2570 versus 3.2572).

**Direct attention-column regularization did suppress the naturally
learned sink.** Muon/S4 went from 37.54% to 9.02% BOS allocation:
a **76.0% relative reduction**. The maximum column share also fell
from 37.66% to 9.18%, providing evidence that the displaced mass was
not simply redirected into one new dominant key. The fraction of
head/layer units with >30% BOS mass went from 66.3% to **0%**.

**The effect was not Muon-specific.** AdamW plus the same penalty
reduced BOS share from 37.49% to 9.83%; the direct penalty is
the apparent active ingredient.

**The quality tradeoff is material.** For Muon/S4 plus penalty,
held-out CE increased by +0.0390 at 512 context and +0.0801
at 1,024 context versus Muon/S4; these regressions appeared in all
three seeds. The corresponding perplexity ratios are approximately
1.040 and 1.083. In addition, mean attention entropy rose from
2.348 to 3.295 nats; the mechanism is largely attention *diffusion*.
Such diffusion can damage legitimate focused retrieval. **It would
be premature to call this a net optimization improvement.**

AdamW without regularization achieved slightly better loss than
Muon under these hyperparameters; these are equal-step comparisons,
not separately tuned or convergence-matched optimizers.

### Acceptance interpretation

- Natural attention sink present without a logit bias: **PASS**.
- Multiple independent training seeds: **PASS (3)**.
- Scale-up to genuine 124M GPT-2 on A100: **PASS**.
- Robust reduction of BOS concentration: **PASS for direct penalty**.
- Exclusion of transfer to another *single* key sink: **PASS on
  maximum-column statistic**, but not a proof of better routing.
- Held-out predictive quality retained without degradation:
  **FAIL**; the regularizer worsened CE, more markedly at 1,024 tokens.
- Schatten-4 Muon initialization independently mitigates sinks:
  **NOT SUPPORTED**.
- From-scratch emergence on natural text: **NOT TESTED**.
- Learned null-attention arm: **NOT TESTED** in this natural-language
  tranche, because GPT-2's attention implementation would require
  a new model architecture and could not share the exact reference
  function at initialization.

This is a strong **mechanistic result with a documented generalization
cost**, not a claim that an unqualified improvement to causal LMs has
been established. The original external-logit-stressor findings
generalize qualitatively to actual sink-bearing pretrained models,
but the two phenomena must not be equated.

## Reproduction and custody

The downloadable evidence bundle contains exact training and corpus
configuration, source code, complete JSONL metrics, per-arm and
per-seed summaries, paired differences, source hash, and SHA-256
checksums. The read-only dataset manifest includes train/validation
token-file hashes but the raw FineWeb text is not republished.
Offline W&B binary telemetry was captured and retained privately on
CHAD, rather than published in GitHub because it can include local
runtime identifiers. Two CI jobs passed on the research branch.

Model weights are not included in this small bundle. Rebuild from
`openai-community/gpt2` and replay the recorded token stream and
optimizer settings.

The scientifically decisive follow-on is a **regularizer-strength
Pareto sweep at matched validation CE**, followed by genuine
from-scratch natural-language pretraining and downstream retrieval
tasks, to determine whether weak regularization can remove sinks
without losing useful sharp attention.
