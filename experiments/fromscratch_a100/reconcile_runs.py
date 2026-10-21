#!/usr/bin/env python3
"""Reconcile an interrupted S4 Colab session with a complete single-arm replay."""
import json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
METRICS=("valid_ce","bos_share","bos_excess","max_column_share",
 "any_key_sink_head_fraction","early_key_sink_head_fraction",
 "sink_head_fraction","attention_entropy","first_five_key_mass",
 "long_valid_ce","long_bos_share","long_max_column_share")

def j(path):
    return json.loads(Path(path).read_text())
def jl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def from_log(name):
    return [json.loads(x[len(name):]) for x in
       (HERE/"compare-live.log").read_text().splitlines() if x.startswith(name)]

def run():
    base=HERE/"evidence/baseline"
    finish=HERE/"evidence/final"
    cb=j(base/"config.json")
    cf=j(finish/"config.json")
    cs=from_log("RUN_CONFIG ")
    assert len(cs)==1
    cs=cs[0]
    assert cb["arms"]==["adamw"]
    assert cs["arms"]==["s4","s4_plus_penalty"]
    assert cf["arms"]==["s4_plus_penalty"]
    for key in ("steps","batch","accum","context","peak_lr","warmup",
                "eval_every","eval_batches","seeds","model",
                "pretraining_checkpoint","gpu","torch_version"):
        assert cb[key]==cs[key]==cf[key],key
    assert cb["seeds"]==[101] and cb["steps"]==4000
    for key in ("train_sha256","valid_sha256","manifest_sha256"):
        assert cb["corpus"][key]==cs["corpus"][key]==cf["corpus"][key],key
    assert j(base/"run_status.json")["complete"]
    assert j(finish/"run_status.json")["complete"]
    b=j(base/"summary.json")
    f=j(finish/"summary.json")
    assert len(b)==len(f)==1 and b[0]["step"]==f[0]["step"]==4000
    assert b[0]["arm"]=="adamw" and f[0]["arm"]=="s4_plus_penalty"
    old=jl(HERE/"compare-live.jsonl")
    s4=[x for x in old if x["arm"]=="s4"]
    partial=[x for x in old if x["arm"]=="s4_plus_penalty"]
    assert len(s4)==17 and s4[-1]["step"]==4000
    assert partial and max(x["step"] for x in partial)<4000
    ini=[jl(base/"initial.jsonl")[0]["model_sha256_sample"],
         jl(finish/"initial.jsonl")[0]["model_sha256_sample"]]
    ini.extend(x["sampled_init_sha256"] for x in from_log("INITIAL "))
    assert len(ini)==4 and len(set(ini))==1
    rawbase=jl(base/"metrics.jsonl")
    rawfinal=jl(finish/"metrics.jsonl")
    assert rawbase[-1]["step"]==rawfinal[-1]["step"]==4000
    metrics={"adamw":b[0]["last"],"s4":s4[-1],"s4_plus_penalty":f[0]["last"]}
    for name,rec in metrics.items():
        assert rec["step"]==4000
        for key in METRICS:
            assert key in rec and math.isfinite(rec[key]),(name,key)

    natural={
        "BOS_ever_above_10pct":any(x["bos_share"]>.1 for x in rawbase),
        "any_column_ever_above_15pct":any(x["max_column_share"]>.15 for x in rawbase),
        "any_head_fraction_ever_above_20pct":any(
             x["any_key_sink_head_fraction"]>.2 for x in rawbase),
        "initial_BOS_share":jl(base/"initial.jsonl")[0]["bos_share"],
        "final_BOS_share":metrics["adamw"]["bos_share"]}
    natural["pronounced_sink_emerged"]=bool(
        natural["BOS_ever_above_10pct"] or
        (natural["any_column_ever_above_15pct"] and
         natural["any_head_fraction_ever_above_20pct"]))
    paired={name:{key:round(metrics[name][key]-metrics["adamw"][key],10)
       for key in METRICS} for name in ("s4","s4_plus_penalty")}
    record={
      "record_type":"GCL_S4_FROM_SCRATCH_MULTISESSION_ADJUDICATION/1",
      "status":"THREE_ARMS_AT_4000_STEPS",
      "arms":["adamw","s4","s4_plus_penalty"],
      "seed":101,"steps":4000,
      "training_token_positions_per_arm":cb["steps"]*cb["batch"]*cb["accum"]*cb["context"],
      "dataset_manifest_sha256":cb["corpus"]["manifest_sha256"],
      "initial_model_sample_sha256":ini[0],
      "pretrained_weights":False,
      "synthetic_logit_bias":False,
      "optimizer":"AdamW only",
      "original_s4_session_disconnected_after_s4_finished":True,
      "interrupted_combined_last_step":max(x["step"] for x in partial),
      "interrupted_combined_excluded":True,
      "combined_replayed_from_initial_weights_to_step_4000":True,
      "emergence_gate":natural,
      "metrics":metrics,
      "paired_differences_vs_adamw":paired,
      "evidence_of_prevention":natural["pronounced_sink_emerged"] and
          metrics["s4_plus_penalty"]["max_column_share"]<
          metrics["adamw"]["max_column_share"],
      "limitations":["one seed","no pronounced baseline sink","no hyperparameter sweep"]
    }
    out=REPO/"docs/data/scratch_s4_20261021_analysis.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    for name,rec in metrics.items():
        print(name,"ce",round(rec["valid_ce"],4),
         "bos",round(rec["bos_share"],5),
         "max",round(rec["max_column_share"],5),
         "long_ce",round(rec["long_valid_ce"],4))
    print("VALIDATED",record["status"],"emergence",natural["pronounced_sink_emerged"])
    return record

if __name__=="__main__":
    run()
