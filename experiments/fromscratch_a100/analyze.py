#!/usr/bin/env python3
"""Strict analysis of matched from-scratch AdamW vs S4 and S4+column."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

KEYS=("valid_ce","valid_ppl","bos_share","bos_excess","max_column_share",
      "any_key_sink_head_fraction","early_key_sink_head_fraction",
      "sink_head_fraction","attention_entropy",
      "first_five_key_mass","long_valid_ce","long_bos_share",
      "long_max_column_share","long_any_key_sink_head_fraction")

def load(folder):
    folder=Path(folder)
    config=json.loads((folder/"config.json").read_text())
    stat=json.loads((folder/"run_status.json").read_text())
    summary=json.loads((folder/"summary.json").read_text())
    rr=[json.loads(x) for x in (folder/"metrics.jsonl").read_text().splitlines()]
    initials=[json.loads(x) for x in (folder/"initial.jsonl").read_text().splitlines()]
    assert stat["complete"] and len(summary)==stat["expected"]
    assert all(s["completed"] and s["step"]==config["steps"] for s in summary)
    return config,stat,summary,rr,initials

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--comparison",required=True)
    ap.add_argument("--output",required=True)
    opts=ap.parse_args()
    cb,sb,b,rb,ib=load(opts.baseline)
    cc,sc,c,rc,ic=load(opts.comparison)
    assert cb["steps"]==cc["steps"]==4000
    assert cb["seeds"]==cc["seeds"]==[101]
    assert cb["arms"]==["adamw"]
    assert cc["arms"]==["s4","s4_plus_penalty"]
    for k in ("context","batch","accum","peak_lr","warmup","eval_batches"):
        assert cb[k]==cc[k],(k,cb[k],cc[k])
    assert cb["corpus"]["train_sha256"]==cc["corpus"]["train_sha256"]
    assert cb["corpus"]["valid_sha256"]==cc["corpus"]["valid_sha256"]
    initial=ib+ic
    assert len(initial)==3
    assert len(set(z["model_sha256_sample"] for z in initial))==1,"init differs"
    final={row["arm"]:row["last"] for row in b+c}
    assert len(final)==3
    baseline=final["adamw"]
    comparisons={}
    for arm in ("s4","s4_plus_penalty"):
        cand=final[arm]
        comparisons[arm]={
         key:{"baseline":baseline[key],"candidate":cand[key],
              "difference":cand[key]-baseline[key]} for key in KEYS
        }
    natural={
      "bos_ever_greater_than_10_percent":any(row["bos_share"]>.10 for row in rb),
      "max_col_ever_greater_than_15_percent":any(row["max_column_share"]>.15 for row in rb),
      "head_fraction_ever_greater_than_20_percent":any(row["any_key_sink_head_fraction"]>.2 for row in rb),
      "absolute_bos_growth_pp":100*(baseline["bos_share"]-ib[0]["bos_share"]),
      "observed_pronounced_sink":bool(any(row["bos_share"]>.10 or
        (row["max_column_share"]>.15 and row["any_key_sink_head_fraction"]>.2)
            for row in rb))}
    outcome={
       "record_type":"GCL_S4_SCRATCH_A100_RESULT/1",
       "status":"COMPLETE",
       "random_initialization":True,"pretrained_weights":False,
       "adversarial_logit_bias":False,"optimizer":"AdamW only",
       "model_params":b[0]["last"]["params"],
       "seed":101,"steps":4000,
       "tokens_per_run":4000*cb["batch"]*cb["accum"]*cb["context"],
       "train_corpus_manifest_sha":cb["corpus"]["manifest_sha256"],
       "exact_same_initial_model_sample_hash":initial[0]["model_sha256_sample"],
       "emergence":natural,"final":final,
       "difference_from_unregularized":comparisons,
       "accepted_prevention_claim":natural["observed_pronounced_sink"] and
         final["s4_plus_penalty"]["max_column_share"]<baseline["max_column_share"],
       "warning":"Only one matched seed and finite pretraining budget; "
                 "lower key-column concentration alone does not prove sink prevention."
    }
    Path(opts.output).write_text(json.dumps(outcome,sort_keys=True,indent=2)+"\n")
    print("EMERGENCE",json.dumps(natural,sort_keys=True))
    for name,measures in final.items():
        print(f"{name:20s} BOS={measures['bos_share']:.4f} "
              f"maxcol={measures['max_column_share']:.4f} "
              f"anyhead={measures['any_key_sink_head_fraction']:.3f} "
              f"CE={measures['valid_ce']:.4f}")
    print("ACCEPTED",outcome["accepted_prevention_claim"])

if __name__=="__main__":main()
