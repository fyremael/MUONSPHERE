#!/usr/bin/env python3
"""Validate and aggregate A100 FineWeb GPT2 natural-sink continuation experiments."""
import argparse,csv,json,math,statistics
from pathlib import Path

ARMS=("muon_frob","muon_s4","muon_s4_conc","adamw","adamw_conc")
KEYS=("valid_ce","bos_real_share","bos_excess","max_column_real_share",
      "head_sink_rate_30","head_sink_rate_50","attention_entropy",
      "long_valid_ce","long_bos_real_share","long_max_column_real_share")
def analysis(root,output):
    root=Path(root)
    cfg=json.loads((root/"config.json").read_text())
    status=json.loads((root/"run_status.json").read_text())
    summary=json.loads((root/"summary.json").read_text())
    records=[json.loads(x) for x in (root/"metrics.jsonl").read_text().splitlines() if x]
    initial=[json.loads(x) for x in (root/"initial_metrics.jsonl").read_text().splitlines() if x]
    seeds=cfg["seeds"];arms=cfg["arms"]
    assert seeds==[101,202,303] and arms==list(ARMS)
    expected={(seed,arm) for seed in seeds for arm in arms}
    assert status["status"]=="complete",status
    assert status["completed"]==status["expected"]==15,status
    assert len(summary)==len(expected)
    assert {(r["seed"],r["arm"]) for r in summary}==expected
    assert {(r["seed"],r["arm"]) for r in initial}==expected
    assert all(r["completed"] and r["steps_completed"]==cfg["steps"] for r in summary)
    assert all(r["bos_excess"]>0.15 and r["head_sink_rate_30"]>0.2 for r in initial)
    # Duplicated initialization genuinely identical across different optimizers
    for seed in seeds:
        b=[r for r in initial if r["seed"]==seed]
        assert max(r["bos_real_share"] for r in b)-min(r["bos_real_share"] for r in b)<1e-6
        assert max(r["valid_ce"] for r in b)-min(r["valid_ce"] for r in b)<1e-6
    final={(r["seed"],r["arm"]):r["last_metrics"] for r in summary}
    for row in summary:
        assert row["last_metrics"]["long_bos_real_share"]>=0
        assert math.isfinite(row["last_metrics"]["valid_ce"])
    means={}
    for arm in arms:
        means[arm]={}
        for key in KEYS:
            vals=[final[(s,arm)][key] for s in seeds]
            means[arm][key]={"mean":statistics.mean(vals),"sd":statistics.stdev(vals),
                            "seeds":dict(zip(seeds,vals))}
    differences={}
    for arm in arms:
        if arm=="muon_s4":continue
        differences[arm]={}
        for key in KEYS:
            vals=[final[(s,arm)][key]-final[(s,"muon_s4")][key] for s in seeds]
            differences[arm][key]={"mean":statistics.mean(vals),
              "sd":statistics.stdev(vals),"seeds":dict(zip(seeds,vals))}
    checks={}
    for method in ("muon_s4_conc","adamw_conc"):
        checks[method]=dict(
          bos_reduced_all_seeds=all(final[(s,method)]["bos_real_share"]<
                final[(s,method.replace("_conc",""))]["bos_real_share"] for s in seeds),
          maxcol_reduced_all_seeds=all(final[(s,method)]["max_column_real_share"]<
                final[(s,method.replace("_conc",""))]["max_column_real_share"] for s in seeds),
          ce_regression_all_seeds=all(final[(s,method)]["valid_ce"]>
                final[(s,method.replace("_conc",""))]["valid_ce"] for s in seeds))
    report={"record_type":"GCL_NATURAL_ATTENTION_SINK_A100_ANALYSIS/1",
      "complete":True,"seeds":seeds,"arms":arms,
      "context":cfg["context"],"accum":cfg["accum"],
      "steps":cfg["steps"],"train_tokens_processed":len(expected)*cfg["steps"]*
        cfg["microbatch"]*cfg["accum"]*cfg["context"],
      "unique_train_corpus_tokens":12000000,
      "model":cfg["pretraining_checkpoint"],
      "model_parameters":124439808,
      "source_natural_sink_gates_passed":len(initial),
      "mean_sd":means,
      "paired_differences_vs_muon_s4":differences,
      "checks":checks,
      "caveat":"continued pretraining of already pretrained GPT2; no random-init sink emergence proven"}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for arm in arms:
        d=means[arm]
        print(f"{arm:20s} BOS={d['bos_real_share']['mean']:.3f} "
              f"max={d['max_column_real_share']['mean']:.3f} "
              f"longBOS={d['long_bos_real_share']['mean']:.3f} "
              f"CE={d['valid_ce']['mean']:.4f}")
    print("CHECKS",json.dumps(checks))
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    analysis(a.root,a.output)
