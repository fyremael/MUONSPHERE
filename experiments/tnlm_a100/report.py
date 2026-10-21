from pathlib import Path
import json
r=Path(__file__).resolve().parents[2]
d=json.loads((r/'docs/data/tnlm_fsm_20261021_analysis.json').read_text())
assert d['status']=='COMPLETE' and d['gate']['fsm_tnlm']['passed']
labels=[('tnlm','TNLM only'),('fsm_tnlm','FSM + TNLM'),('fsm_tnlm_s4','FSM + S4'),('fsm_tnlm_s4_col','FSM + S4 + column')]
rows=[]
for arm,label in labels:
 x=d['final'][arm]
 rows.append(f"| {label} | {100*x['bos_share']:.2f}% | {100*x['strongest_layer_sink_fraction']:.2f}% | {100*x['max_column_share']:.2f}% | {x['valid_ce']:.4f} | {x['long_valid_ce']:.4f} |")
gate=d['gate']['fsm_tnlm']
a=d['final']['fsm_tnlm'];b=d['final']['fsm_tnlm_s4_col'];c=d['final']['fsm_tnlm_s4']
md="""# A100 TNLM / Force-sink induction and Schatten-4 mitigation

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
"""+'\n'.join(rows)+f"""

The sustained induction gate requires at least 25% of heads in the same
layer to exceed 30% BOS attention, for three consecutive evaluations.
TNLM alone failed that criterion; FSM/TNLM passed (first qualifying
window ending at step {gate['steps'][-1]}, layer {gate['layer']+1}).

The combined regularizer reduced BOS attention by
{100*(1-b['bos_share']/a['bos_share']):.1f}% relative to FSM-only, with
CE512 difference {b['valid_ce']-a['valid_ce']:+.4f}.
For S4 only, BOS share was {100*c['bos_share']:.2f}%.

The original FSM run was interrupted and its incomplete trace excluded;
the completed replay is used. The first S4-only controller failed
during unnecessary dataset preparation; the direct run used independently
checksum-verified cached token files. The failure receipt remains
preserved, and the completed run was separately checked.

Scientific scope: one paired seed, structural induction, no proof of
naturally spontaneous pathological sinks or broad downstream benefit.
Reproducibility is recorded in the evidence bundle.
"""
p=r/'docs/attention_sink_tnlm_fsm_20261021_report.md'
p.write_text(md)
print('REPORT',p, 'chars',len(md))
