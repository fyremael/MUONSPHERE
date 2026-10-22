# Natural-language attention sinks on A100: GPT-2 124M

## Experimental question and scope

The synthetic 10.7M-parameter experiment showed that direct fourth-order
attention-column regularization can suppress a deliberately injected
BOS-key logit sink. That result did not establish behavior under genuine
language-model attention.

This follow-on evaluates **naturally learned attention sinks with zero
logit injection** using a genuine pretrained GPT-2 Small (124,439,808
parameters) checkpoint, and continues autoregressive training on
FineWeb-Edu. It tests **persistence and reversal of pre-existing sinks**;
it does **not** measure the spontaneous birth of sinks from randomly
initialized weights. Claims about de novo emergence require pretraining
from scratch on substantially more text.

## Corpus and provenance

- Dataset: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`;
  streaming source as of 2026-10-21.
- Tokenizer: tiktoken GPT-2 BPE; 50,257 vocabulary.
- Cached 12,000,000 train tokens and 1,000,000 validation tokens.
- Per-document disjoint split: SHA-256 of dataset `id`, first
  eight hex digits modulo 11; bucket zero validation, others training.
- Documents are concatenated with GPT-2 EOT token 50256; the start of
  every sample is explicitly EOT, which is appropriate for GPT-2's
  start/end-of-text convention. No attention-logit manipulation.
- Validation windows use a fixed RNG (71289) and do not change across
  optimizer arms or random-seed conditions.
- Training batches use per-seed fixed random offsets into the same
  document-disjoint training token stream.
- The corpus manifest records hashes for the document-id provenance
  and the exact token files. The public repo omits the dataset texts
  and large token binaries; fetch from source to reproduce.

## Model, controls and optimizer

Reference checkpoint: `openai-community/gpt2` (124.44M parameters,
12 layers, embedding width 768, 12 attention heads, learned positional
embeddings, maximum 1,024-token context). All arms start from this
**identical pretrained checkpoint** and use dropout probability zero.
The model weights remain full-precision; forward uses BF16 autocast
on A100 40GB; Muon Newton–Schulz uses FP32. Original GPT-2
attention is `eager` in the arm needing attention probabilities.

Pretraining objective: ordinary causal next-token cross entropy over
FineWeb-Edu sequences. No copying-task auxiliary loss.

Experimental arms:
1. `muon_frob`: Muon with Frobenius initialization.
2. `muon_s4`: identical Muon but Schatten-4 initialization.
3. `muon_s4_conc`: Muon S4 plus directly differentiable
   fourth-order positive-excess attention-column energy.
4. `adamw`: AdamW continuation-pretraining control.
5. `adamw_conc`: AdamW with the **same** attention regularizer,
   separating the effect of the penalty from the optimizer.

The fourth-order attention penalty is **not a Schatten-4 matrix norm**.
It takes attention probabilities for queries in the final 75% of each
layer, subtracts the causal-uniform allocation, then penalizes the
squared sum of per-key squared attention excess. That penalty is applied
to the *attention probabilities*, not to Muon's singular values.

The Muon implementation is a self-contained 2D-weight reference
implementation; it only updates 2D attention/MLP block matrices.
Embeddings, norm scales, biases and output head follow AdamW.
Other schemes are not tuned to compute-equal or loss-equal optima.

## Natural sink measurements

Evaluation records:
- BOS real-token attention share, averaged over later queries, heads
  and layers; because standard GPT-2 attention sums to 1 this equals
  absolute BOS mass.
- BOS mass minus the causal uniform expected first-key probability.
- Max real-key column share over late queries. This catches migration
  from BOS to another preferred token.
- Fraction of heads where late-query BOS mass exceeds 30% and 50%.
- Attention entropy and held-out language-model cross entropy.
- A 1,024-token held-out window check at the final step for each arm.

The **pretrained model must pass a natural sink gate**:
BOS excess >0.15 and >20% of head/layer units with >30% BOS
attention, before any additional training. No synthetic logit bias is
used anywhere. If the condition fails, the run stops rather than
reporting artificial sink suppression.

## A100 launch

A single A100-SXM4-40GB Colab session is allocated, corpus prepared,
and training dispatched using the authenticated Colab CLI. Example
run arguments (source: `train_natural.py`):

```bash
python train_natural.py \
  --data /content/gcl_natural_sink \
  --out /content/gcl_natural_sink_full \
  --context 512 --microbatch 2 --accum 2 \
  --steps 320 --eval-every 40 --eval-batches 4 \
  --seeds 101 202 303 \
  --arms muon_frob muon_s4 muon_s4_conc adamw adamw_conc \
  --lambda-conc 0.03 --wandb-mode offline \
  --require-natural-sink --eval-long-context
```

320 steps × 2 × 2 × 512 = 655,360 training tokens per
seed/arm; up to 9,830,400 continuation-pretraining tokens over
15 distinct arms at equal **optimizer-step and input-token** budgets.
Validation is on held-out documents.

This is a significant scale-up from the synthetic test, but remains
small in comparison with multi-billion-token training from scratch.
A convergence-matched comparison requires LR/regularizer sweeps,
longer runs and additional model sizes.

## Completion and interpretation

Record `config.json`, `initial_metrics.jsonl`,
`metrics.jsonl`, `summary.json`, `run_status.json`,
A100 model and dataset metadata. Persist offline W&B records locally.
Do not mark this result admitted or positive until all seed/arm results
are checked for completeness, held-out loss regression, and whether
max-column concentration moved to another token.
