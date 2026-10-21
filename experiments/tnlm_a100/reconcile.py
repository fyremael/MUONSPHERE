#!/usr/bin/env python3
"""Reconcile interrupted induction, full FSM replay, S4 matched runs."""
import json,math,argparse,hashlib
from pathlib import Path
from collections import defaultdict

HERE=Path(__file__).resolve().parent
METRICS=("valid_ce","bos_share","bos_excess","max_column_share",
         "sink_head_fraction","strongest_layer_sink_fraction",
         "any_key_sink_head_fraction","attention_entropy",
         "long_valid_ce","long_bos_share","long_max_column_share")
STEPS=4000
MANIFEST="5233f44b9091a9d4f0af9f5abebfc6bcb64d4da098f7cfddf52bfe91630acbba"

def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def log(label):
    return [json.loads(x[len(label):]) for x in (HERE/"induction-run.log").read_text().splitlines()
            if x.startswith(label)]
def stable_gate(rows):
    valid=sorted([r for r in rows if r["step"]>1],key=lambda r:r["step"])
    for layer in range(12):
        for start in range(len(valid)-2):
            seq=valid[start:start+3]
            if all(x["sink_fraction_by_layer"][layer]>=.25 for x in seq):
                return {"passed":True,"layer":layer,
                        "steps":[x["step"] for x in seq],
                        "fractions":[x["sink_fraction_by_layer"][layer] for x in seq]}
    return {"passed":False,
      "best_layer_fraction":max(max(x["sink_fraction_by_layer"]) for x in valid)}
def load_completed(prefix,arm):
    path=HERE
    status=json.loads((path/(prefix+"-controller-status.json")).read_text())
    assert status["status"]=="COMPLETE",status
    cfg=json.loads((path/(prefix+"-config.json")).read_text())
    assert cfg["arms"]==[arm] and cfg["steps"]==STEPS
    assert cfg["corpus"]["manifest_sha256"]==MANIFEST
    assert cfg["seeds"]==[101]
    rows=jl(path/(prefix+"-metrics.jsonl"))
    assert rows[-1]["step"]==STEPS
    summary=json.loads((path/(prefix+"-summary.json")).read_text())
    assert len(summary)==1 and summary[0]["completed"]
    ini=jl(path/(prefix+"-initial.jsonl"))
    assert len(ini)==1
    assert summary[0]["initial_sha256"]==ini[0]["model_sha256_sample"]
    return cfg,rows,summary[0]["last"],ini[0]

def analyze(allow_missing=False):
    original=jl(HERE/"induction-metrics.jsonl")
    tn=[r for r in original if r["arm"]=="tnlm"]
    first=[r for r in original if r["arm"]=="fsm_tnlm"]
    assert tn and tn[-1]["step"]==STEPS
    assert first and first[-1]["step"]<STEPS
    cl=log("RUN_CONFIG ")
    assert len(cl)==1
    assert cl[0]["corpus"]["manifest_sha256"]==MANIFEST
    initials=log("INITIAL ")
    original_sha={r["sampled_init_sha256"] for r in initials}
    assert len(original_sha)==1
    series={"tnlm":tn}
    final={"tnlm":tn[-1]}
    completed={}
    expected={"replay":"fsm_tnlm","s4col":"fsm_tnlm_s4_col",
              "s4only":"fsm_tnlm_s4"}
    for key,arm in expected.items():
        path=HERE/(key+"-controller-status.json")
        if not path.is_file():
            if key!="s4only" and not allow_missing:
                raise AssertionError(f"{key} not completed")
            continue
        cfg,rows,last,initial=load_completed(key,arm)
        for k in ("context","batch","accum","warmup","peak_lr",
                  "eval_every","eval_batches","seeds","steps"):
            assert cfg[k]==cl[0][k],(key,k)
        assert initial["model_sha256_sample"] in original_sha
        completed[key]={"arm":arm,"steps":last["step"],
                        "full_valid_ce":last["valid_ce"]}
        series[arm]=rows
        final[arm]=last
    assert "fsm_tnlm" in series,series.keys()
    gate={name:stable_gate(data) for name,data in series.items()}
    assert gate["fsm_tnlm"]["passed"],gate
    differences={}
    for arm,row in final.items():
        if arm not in ("tnlm","fsm_tnlm"):
            differences[arm]={k:row[k]-final["fsm_tnlm"][k]
                             for k in METRICS}
    report={"record_type":"GCL_TNLM_S4_A100_RECONCILIATION/1",
            "status":"COMPLETE" if "s4col" in completed else "INCOMPLETE",
            "seed":101,"random_initialization":True,
            "optimizer":"AdamW","no_injected_attention_logit":True,
            "corpus_document_hash":MANIFEST,
            "model_initial_sample_sha256":next(iter(original_sha)),
            "training_token_positions_per_arm":STEPS*cl[0]["batch"]*
                                              cl[0]["accum"]*cl[0]["context"],
            "original_interrupted_fsm_step":first[-1]["step"],
            "original_incomplete_trace_excluded":True,
            "completed_arms":completed,
            "gate":gate,
            "final":final,
            "paired_differences_vs_fsm":differences,
            "interpretation":"FSM structurally favors sink; not a spontaneous baseline"}
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--allow-missing",action="store_true")
    p.add_argument("--output")
    opts=p.parse_args()
    report=analyze(opts.allow_missing)
    if opts.output:
        out=Path(opts.output)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("ADJUDICATION",report["status"],report["completed_arms"])
    for name,metrics in report["final"].items():
        print(name,{"step":metrics["step"],
          "bos_share":round(metrics["bos_share"],4),
          "strongest_layer":round(metrics["strongest_layer_sink_fraction"],4),
          "max_column":round(metrics["max_column_share"],4),
          "CE":round(metrics["valid_ce"],4)})
