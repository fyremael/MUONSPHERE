# A100 TNLM / Force-sink induction and Schatten-4 mitigation

Random-init GPT-2 Small (124M), FineWeb-Edu sample-10BT, AdamW, seed 101.
All arms processed 16,384,000 token positions in 4,000 matched updates.
No pretrained checkpoint and no attention-logit bias.

TNLM masks the prediction made from position zero. Force-sink Masking
(FSM) reserves 16 residual channels for the position-zero mechanism.
FSM is a deliberately structurally biased induction control, not
spontaneous sink formation in an unchanged transformer.

Schatten-4 regularizes the exact fourth-power norm of the deviation
from causal uniform attention on 128 sampled query rows per head.
The combined arm adds fourth-order positive-excess column energy.

| Arm | BOS share | Strongest-layer sink heads | Max column | CE512 | CE1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TNLM only | 3.47% | 8.33% | 6.00% | 5.5976 | 5.8904 |
| FSM + TNLM | 10.82% | 37.50% | 12.58% | 5.5828 | 5.8822 |
| FSM + S4 | 5.24% | 25.00% | 6.71% | 5.5835 | 5.8806 |
| FSM + S4 + column | 4.51% | 14.58% | 5.99% | 5.5858 | 5.8969 |

The sustained induction gate requires at least 25% of heads in the same
layer to exceed 30% BOS attention, for three consecutive evaluations.
TNLM alone failed that criterion; FSM/TNLM passed (first qualifying
window ending at step 2000, layer 12).

The combined regularizer reduced BOS attention by
58.3% relative to FSM-only, with
CE512 difference +0.0030.
For S4 only, BOS share was 5.24%. S4-only still
passed the sustained sink gate, with 25.0% of heads in the strongest
layer exceeding 30% BOS attention at the final checkpoint. The
combined arm did not pass and ended at 14.58% in its strongest layer.
The targeted column penalty therefore adds a measurable benefit
beyond S4 alone under this structural induction. It is not evidence
of better downstream retrieval.

The original FSM run was interrupted and its incomplete trace excluded;
the completed replay is used. The first S4-only controller failed
during unnecessary dataset preparation; the direct run used independently
checksum-verified cached token files. The failure receipt remains
preserved, and the completed run was separately checked.

Scientific scope: one paired seed, structural induction, no proof of
naturally spontaneous pathological sinks or broad downstream benefit.
Reproducibility is recorded in the evidence bundle.
