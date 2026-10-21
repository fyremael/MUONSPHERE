# A100: From-scratch natural-text attention sink emergence

This experiment removes Muon completely. Every arm uses the same AdamW
optimizer and a GPT-2 Small (124M parameter) model initialized randomly,
not a pretrained checkpoint. The corpus is FineWeb-Edu sample-10BT,
with 60 million training tokens and 4 million held-out tokens and a
document-disjoint deterministic hash partition. GPT-2 BPE tokenization
is used. The training sequence begins with the standard GPT-2 EOT
start-of-document symbol. No synthetic logit bias is introduced.

Three comparison arms:

1. adamw: causal language model, no attention regularization.
2. s4: exact Schatten-4 fourth-power penalty on the attention matrix
   deviation from the causal uniform distribution. The matrix includes
   128 deterministic late-query rows and all key columns. The penalty
   is tr((D D^T)^2)/Q^2 per head, averaged across heads and layers.
3. s4_plus_penalty: the same true Schatten-4^4 penalty plus the
   fourth-order positive-excess key-column energy penalty.

All arms use the same random model initialization for each seed, the
same training window offsets, token budget, AdamW hyperparameters,
validation windows and learning-rate schedule. BF16 forward is used on
the A100; the Schatten-4 Gram computation uses FP32.

Diagnostics include BOS attention share, excess above causal uniform,
maximum attention share awarded to any column, fractions of heads with
BOS or any-key concentration exceeding 30%, whether the peak key
is among the first five tokens, held-out cross-entropy, perplexity,
attention entropy and 1024-token context tests.

The scientific acceptance condition is explicit: an unregularized
random-init baseline must first exhibit naturally emerging sink
concentration. If no baseline sink forms within the training budget,
do NOT call suppressed attention in the penalty arm mitigation.

Training source: train_scratch.py. Corpus preparation:
prepare_fineweb.py. A short baseline pilot ran 320 steps with seed 101
and did not produce a BOS sink (final BOS share 0.00194; maximum key
column share approximately 0.046; held-out CE about 6.854).

A longer baseline run is therefore required to test emergence and a
controlled matched-seed ablation is required to assess prevention.
