"""Generate a plain-language, number-for-number replication and retrieval report."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
d=json.loads((REPO/'docs/data/tnlm_fsm_three_seed_retrieval_analysis.json').read_text())
assert d['status']=='COMPLETE'
arms=(('fsm_tnlm','FSM baseline'),
      ('fsm_tnlm_s4','FSM + S4'),
      ('fsm_tnlm_s4_col','FSM + S4 + column'))
fmt=lambda x: f'{x:.4f}'
pct=lambda x:f'{x*100:.2f}%'
rows=[]
for arm,label in arms:
    m=d['stats'][arm]
    count=sum(d['gates'][str(seed)][arm]['passed'] for seed in d['seeds'])
    rows.append('| '+label+' | '+pct(m['bos_share']['mean'])+
        ' ± '+pct(m['bos_share']['sd'])+' | '+
        pct(m['strongest_layer_sink_fraction']['mean'])+' | '+
        f'{count}/3 | '+fmt(m['valid_ce']['mean'])+' | '+
        fmt(m['long_valid_ce']['mean'])+' |')
retrows=[]
for arm,label in arms:
    for length in ('512','1024'):
        m=d['stats'][arm]['retrieval'][length]
        retrows.append('| '+label+' | '+length+' | '+
            fmt(m['natural_copy_lift_nats']['mean'])+' ± '+
            fmt(m['natural_copy_lift_nats']['sd'])+' | '+
            pct(m['natural_copy_top1']['mean'])+' | '+
            pct(m['passkey_fourway_correct']['mean'])+' | '+
            fmt(m['passkey_ablation_true_logit_lift']['mean'])+' |')
baseline=sum(d['gates'][str(seed)]['fsm_tnlm']['passed'] for seed in d['seeds'])
combined=sum(d['gates'][str(seed)]['fsm_tnlm_s4_col']['passed'] for seed in d['seeds'])
s4=sum(d['gates'][str(seed)]['fsm_tnlm_s4']['passed'] for seed in d['seeds'])
p=d['paired_differences']
doc=f"""# Three-seed replication and downstream retrieval probes

**Completed experiment**: GPT-2 Small 124M from random initialization,
trained on FineWeb-Edu natural text using AdamW (no Muon) and deliberately
structured Force-sink Masking plus the TNLM loss. No synthetic logit bias
and no pretrained checkpoints. Seeds 101, 202 and 303 were run through
all three matched arms (9 full 4,000-step runs), at 512 context,
effective batch 8 and identical 16,384,000 processed token positions
per run. The dataset contains 60 million train and 4 million validation
tokens, document-separated with identical SHA-256 token-file hashes.

Two A100-SXM4 40 GB GPUs were used. Shared use of one A100 by two
isolated seed processes raised recorded utilization to 99% while using
11 GB GPU RAM, without altering training hyperparameters. This is a
matched replication study, **not a larger-model scaling experiment**.

## Sustained attention sinks

Induction was accepted only when at least 25% of heads in the same
transformer layer gave BOS over 30% of attention for three successive
validation checkpoints. Counts below report the number of seeds meeting
that *same* stringent criterion. Mean BOS percentages carry sample SD
across independent seeds.

| Method | BOS share (mean ± SD) | Strongest-layer head fraction | Sustained sink seeds | Held-out CE512 | CE1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The structurally induced baseline crossed the sustained gate in
**{baseline}/3** seeds. S4 alone crossed it in **{s4}/3** seeds;
S4 plus the targeted column penalty crossed it in **{combined}/3**
seeds. These are single-configuration findings; do not extrapolate
to unconstrained transformers that never formed the sink.

Paired mean change versus the forced-sink baseline:
- S4-only BOS share: {pct(p['fsm_tnlm_s4']['bos_share']['mean'])};
  held-out CE512: {p['fsm_tnlm_s4']['valid_ce']['mean']:+.4f}.
- S4 + column BOS share: {pct(p['fsm_tnlm_s4_col']['bos_share']['mean'])};
  held-out CE512: {p['fsm_tnlm_s4_col']['valid_ce']['mean']:+.4f}.
- Combined arm 95% three-seed t interval for BOS-share change:
  [{pct(p['fsm_tnlm_s4_col']['bos_share']['CI95'][0])},
   {pct(p['fsm_tnlm_s4_col']['bos_share']['CI95'][1])}].

## Downstream retrieval under intact-versus-ablated context

The model was tested **without retrieval fine-tuning** on deterministic
held-out stimuli at 512 and 1,024 positions. The natural passage-copy
probe repeats eight validation-corpus tokens near the end of a
sequence; it scores the prediction with those tokens earlier in the
context versus the same prompt with earlier evidence replaced by a
different passage. Positive \`copy_lift\` means *lower* NLL when the
relevant earlier span exists. It must be materially positive to claim
the model actually uses long-range context.

A four-choice passkey test is included with a fresh passkey near the
beginning of natural-text filler and a query near the end.
**Chance accuracy is 25%**. A context-ablated value variant measures
sensitivity to the passkey rather than changes in general logits.

| Method | Context | Copy lift (nats/token, mean ± SD) | Span top-1 token accuracy | 4-choice passkey accuracy | Passkey-ablation logit lift |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(retrows)}

**Interpretation:** These are zero-shot context-use probes, not
supervised retrieval benchmarks. Four-choice results near 25%, or
negligible intact-vs-ablated differences, mean no reliable retrieval
competence has been demonstrated. Suppressing an attention sink is
not automatically improvement in downstream retrieval. Claims of
positive retrieval transfer require both useful accuracy and
context-sensitive ablation lift that replicate across seeds.

## Verification and constraints

The strict three-seed analyzer verifies each of nine completed
4,000-step training arms, exact matching train/validation corpus
hashes, identical initial weights across arms within each seed,
distinct initial fingerprints between seeds, held-out validation
results at 512 and 1024, and the full retrieval records. It computes
paired seed-specific differences and t intervals using 3 seeds
(df=2, wide intervals). Per-seed results, input source, W&B receipt
custody, GPU utilization and SHA-256 manifest are in the public
evidence package; offline W&B binaries are retained privately.

The corpus is educational natural-language text but the retrieval
questions are controlled constructed prompts. This is a deliberate
causal probe for use of earlier evidence. The 124M LM has seen a
relatively modest pretraining budget, and passkey behavior may remain
at floor regardless of regularization. The clean follow-on is
retrieval-trained or substantially longer-pretrained models, while
preserving the same paired sink interventions.

**Scientific boundary:** Force-sink Masking establishes a structural
positive control; neither this experiment nor its seed replication
proves sink prevention in an otherwise unmodified transformer.
"""
out=REPO/'docs/attention_sink_tnlm_fsm_three_seed_retrieval_report.md'
out.write_text(doc)
print("FINAL_REPORT",out,len(doc))

