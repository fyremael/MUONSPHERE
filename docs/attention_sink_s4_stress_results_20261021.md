# Schatten-4 attention-sink stress: completed three-seed results

**Execution:** 2026-10-21 UTC; authenticated Colab CLI, Tesla T4;
PyTorch 2.11.0+cu130; 6-layer / width-384 / 8-head causal transformer,
10,684,416 non-null parameters; 120 optimizer steps per arm, batch 16,
24-symbol repeated-span causal language modeling, 3 seeds (101, 202, 303);
six optimizer/attention arms. All 18 runs completed. The +7.0 BOS-key
logit stressor cleared 18/18 initial positive-control gates;
minimum initial induced-vs-clean BOS mass increase **0.9348** and
minimum induced BOS real-token share **0.9632**.

## Aggregated outcomes

Means ± sample standard deviations across the three seeds.

| Method | BOS real share | Highest column real share | Source-key real share | Copy accuracy | Held-out CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | .8359 ± .0391 | .9256 ± .0128 | .0086 ± .0049 | .0304 ± .0023 | 3.3524 ± .0014 |
| Muon/Frobenius | .7007 ± .0136 | .7202 ± .0102 | .0741 ± .0071 | .9991 ± .0015 | 1.6859 ± .0115 |
| Muon/S4 | .6994 ± .0145 | .7190 ± .0105 | .0744 ± .0064 | .9988 ± .0013 | 1.6836 ± .0099 |
| Muon/S4 + column penalty | .0070 ± .0018 | .0643 ± .0006 | .1039 ± .0109 | .9870 ± .0031 | 1.7088 ± .0044 |
| Muon/S4 + null | .7013 ± .0433 | .7069 ± .0423 | .0778 ± .0021 | .9988 ± .0010 | 1.6839 ± .0049 |
| Muon/S4 + both | .0069 ± .0013 | .0622 ± .0011 | .1013 ± .0065 | .9878 ± .0038 | 1.7011 ± .0042 |

### Interpretation

1. **Schatten-4 initialization alone did not suppress the induced
   sink.** Frobenius and S4 Muon were nearly indistinguishable on
   the primary sink metrics at this training budget.
2. **Direct fourth-order column regularization suppressed induced
   concentration**, reducing BOS's share of *real-token* attention
   from 69.94% to 0.70%; the highest other key-column share also fell
   from 71.90% to 6.43%, so the displaced probability was not simply
   relocated to one competing key.
3. **Useful source attention increased:** correct source-key share
   rose from 7.44% to 10.39%. This suggests attention was redirected
   toward task-relevant content, not indiscriminately flattened.
4. **There was a tradeoff:** held-out CE increased by about +.0252,
   while copying accuracy declined by about 1.19 percentage points.
   These are nonzero differences; regularization is not free.
5. **Null attention alone was insufficient** under a fixed +7 key-logit
   perturbation. The null arm used only about 0.44% null probability
   at convergence, with BOS's conditional real-token share still ~70%.
6. **Counterfactual sensitivity:** removing BOS access after training
   increased CE by +2.039 for Muon/S4, +.547 with column penalty,
   and only +.060 with both penalty and null attention. The combined
   method was less reliant on the BOS route, although this ablation
   changes all layers simultaneously.
7. AdamW reached only ~3% copy accuracy in 120 steps; thus it is an
   **undertrained reference, not a fair convergence-matched comparator**.
   Muon versus AdamW superiority cannot be inferred from this setup.

**Conclusion:** This supports robustness of direct fourth-order
attention-column regularization to an *externally injected*
single-key logit sink. It does not establish that the same method
prevents the natural formation of attention sinks in pretrained LMs.

### Adversarial diagnostic validation

Independent Colab hostile metric controls passed:

- A sink **moved to key 1** had BOS share ~1.8% but highest column
  share ~98.2%; a BOS-only statistic would falsely declare success.
- A head with a dominant learned **null logit** showed apparent BOS
  mass ~0.034%, while conditional BOS share remained ~98.8% and null
  mass was ~99.97%; unconditional BOS mass would falsely declare success.
- Uniform causal attention had BOS conditional share ~2.73% and no
  fourth-order excess, as expected.

These are deterministic diagnostic tests, not further training seeds.

### Evidence, reproducibility and limits

Full per-step metrics, config, induction-gate records, calibration,
positive-control outputs, by-seed final metrics, aggregated means and
paired differences, and byte-exact Colab run source are supplied in the
downloadable experiment bundle. W&B was run offline in Colab, and its
unredacted diagnostic logs are intentionally not posted to this public
GitHub PR; local CHAD run captures remain available privately.

This uses a copying benchmark with 35 symbols, a forced +7 BOS bias,
three seeds, fixed optimizer hyperparameters and equal steps.
Statistical uncertainty is large with n=3. No confidence intervals
or claims of statistical significance are warranted. The experiment
is not compute-matched, nor does it demonstrate naturally emerging
sinks in open-text language modeling. A credible external-validity
test requires a natural-language pretraining corpus, more model sizes,
long-context generalization, and evaluations at matched validation CE.
