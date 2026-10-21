# True Schatten-4 attention regularization from scratch on FineWeb-Edu

**Completed:** 21 October 2026. Three AdamW-controlled arms, each trained for 4,000 optimizer updates on NVIDIA A100-SXM4-40GB. All comparisons use random initialization, without Muon, without a pretrained checkpoint and without a synthetic attention-logit bias.

## Question and experimental design

Does a genuine Schatten-4 spectral penalty on attention, alone or combined with an additional fourth-order key-column penalty, prevent attention sinks emerging during natural-language autoregressive training?

The model is GPT-2 Small (124,439,808 parameters; 12 layers, 12 attention heads, hidden width 768). The corpus is FineWeb-Edu sample-10BT tokenized with GPT-2 BPE. It uses 60 million training tokens and 4 million held-out tokens, separated at document level by a deterministic SHA-256 partition. Each model receives an EOT token at the beginning of sampled sequences, as is normal for GPT-2-style training. The three arms share a fixed seed (101), verified matching sampled initial parameter hash, document split, input window sequence, AdamW hyperparameters, learning-rate schedule and evaluation samples. Each processed 16,384,000 token positions.

Arms: unregularized AdamW, AdamW plus true sampled-deviation Schatten-4 fourth-power, and AdamW plus that Schatten-4 term and a fourth-order positive-excess key-column penalty. Both regularization coefficients are 0.15.

The S4 loss is exactly trace((D D-transpose)^2) / Q^2, where D is the attention probability matrix minus its causal uniform reference, restricted to 128 evenly spaced later query positions with all key columns retained. This is a true Schatten-4 fourth-power on the resulting submatrix. It is *not* the S4 normalization used in Muon and *not* the Schatten-4 norm of the entire full-sequence attention matrix.

The implementation passed numerical SVD equivalence, null deviation, sink relocation and nonzero gradient tests on the A100.

## Final results

| Intervention | BOS attention | Max key-column attention | Heads with any key above 30% | CE at 512 | CE at 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW baseline | 2.053% | 4.773% | 2.431% | 5.6052 | 5.9104 |
| AdamW + S4 | 1.757% | 4.020% | 0.521% | 5.5977 | 5.9031 |
| AdamW + S4 + column penalty | 1.650% | 3.820% | 0.868% | 5.5916 | 5.9101 |

Adding both regularizers reduced maximum column share by approximately 20% relative to the baseline and lowered held-out cross-entropy at context 512 by 0.0136. At context 1024, the loss was effectively unchanged. The plain S4 arm slightly outperformed the combination on the 1024-token validation loss and exhibited fewer extremely concentrated heads (0.521% versus 0.868%). It is therefore not accurate to claim that the combined penalty dominates S4 alone on every diagnostic.

## Natural emergence gate: NOT MET

Initially BOS key share was 0.365%; at step 4,000 in the unregularized model it was 2.053%. A modest first-key preference appeared, but broad pathological sink formation did not occur. The baseline finished with only 2.431% of heads giving any key more than 30% of attention, and a 4.773% mean maximum-column share. At no measured training checkpoint did BOS share exceed 10%, or maximum column share exceed 15%.

Therefore, the principal proposed assertion, *prevention of a naturally emerging pathological attention sink*, is **not established** by this experiment. The data instead support a modest reduction in attention concentration with negligible or slightly favorable next-token cross-entropy in this one-seed limited-training regime.

## Reproducibility and interruption reconciliation

The baseline and S4-only arms each completed the full 4,000 steps before the original comparison Colab session was lost during the combined arm at step 2,250. An independent replacement A100 was provisioned, the dataset regenerated with matching SHA-256 hashes, and the combined arm restarted from the identical seed and initial weights. That replay completed its full 4,000 steps. Only its completed run is included in the final comparison; the failed partial trace is retained separately as operational evidence.

The repository evidence bundle includes complete per-step measurements for each completed arm, exact configurations, corpus hashes, the S4 mathematical tests, reconstructed executed source, a reconciliation ledger and per-file SHA-256 hashes. The final replay's offline W&B binary and full receipt were retrieved privately before its GPU was released. The previous session's unexported W&B data cannot be attested as retained.

## Limits and next mathematical obligation

This is **one paired random seed**, 16.38 million processed token positions per arm, and a single penalty strength, not a statistically replicated emergence study or an optimized Pareto comparison. Preventing sinks requires demonstrating that the unregularized baseline actually forms reproducible pathological sinks on natural text; then evaluating prevention at matched training budgets, validation loss, longer contexts and downstream information retrieval, ideally across several random seeds and more training tokens. No affirmative prevention claim is admitted here.
