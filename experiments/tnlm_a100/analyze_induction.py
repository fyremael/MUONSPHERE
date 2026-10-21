#!/usr/bin/env python3
"""Exact-head analysis of true natural-text TNLM P0 sink induction.

Reports pilot status without fabricating missing arms; requires three
consecutive evaluation checkpoints in the *same layer* at >=25% of
heads whose BOS mean attention exceeds 30%.
"""
import argparse,json,statistics
from pathlib import Path
from collections import defaultdict

ARMS=("tnlm","fsm_tnlm")
THRESHOLD=0.25
CONSECUTIVE=3

def parse(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def gate(records):
    seq=sorted([r for r in records if r["step"]>1],key=lambda r:r["step"])
    if not seq:return {"passed":False,"reason":"no_evaluation_yet"}
    layers=len(seq[0]["sink_fraction_by_layer"])
    assert layers==12
    best=[]
    windows=[]
    for layer in range(layers):
        streak=0
        for row in seq:
            share=row["sink_fraction_by_layer"][layer]
            if share>=THRESHOLD:streak+=1
            else:streak=0
            if streak>=CONSECUTIVE:
                windows.append({"layer":layer,"end_step":row["step"],
                                "consecutive":streak,"fraction":share})
        best.append(max(r["sink_fraction_by_layer"][layer] for r in seq))
    return {"passed":bool(windows),
            "first_success":min(windows,key=lambda x:x["end_step"]) if windows else None,
            "max_layer_head_fraction":max(best),
            "max_bos_share":max(r["bos_share"] for r in seq),
            "max_column_share":max(r["max_column_share"] for r in seq),
            "latest_step":seq[-1]["step"],
            "latest_full_valid_ce":seq[-1]["valid_ce"],
            "earliest_three_stable_checkpoint":CONSECUTIVE,
            "threshold_head_fraction":THRESHOLD}

def build(metrics,done=None,historical=None):
    rows=parse(metrics)
    grouped=defaultdict(list)
    for r in rows:grouped[(r["seed"],r["arm"])].append(r)
    result={"record_type":"GCL_TNLM_SINK_INDUCTION_STATUS/1",
            "complete":False,"gate_definition":{
              "threshold_BOS_prob_per_head":0.30,
              "fraction_of_heads_within_one_layer":THRESHOLD,
              "consecutive_eval_checkpoints":CONSECUTIVE},
            "run_count":len(grouped),"arms":{}}
    for (seed,arm),arm_rows in grouped.items():
        a=gate(arm_rows)
        a["records"]=len(arm_rows)
        a["seed"]=seed
        a["arm"]=arm
        result["arms"][arm]=a
    if done and Path(done).is_file():
        status=json.loads(Path(done).read_text())
        assert status.get("count")==status.get("expected"),status
        result["complete"]=bool(status.get("complete"))
        result["final_status"]=status
        assert set(result["arms"])==set(ARMS)
        assert all(result["arms"][a]["latest_step"]==4000 for a in ARMS)
    if historical and Path(historical).is_file():
        baseline=parse(historical)
        base={r["step"]:r for r in baseline}
        result["historical_baseline"]={
          "seed":101,
          "step_4000_BOS":base[4000]["bos_share"],
          "step_4000_CE":base[4000]["valid_ce"],
          "gate_observed":any(r["bos_share"]>0.10 for r in baseline)}
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--metrics",required=True)
    p.add_argument("--status")
    p.add_argument("--baseline")
    p.add_argument("--output")
    a=p.parse_args()
    report=build(a.metrics,a.status,a.baseline)
    if a.output:
        Path(a.output).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
