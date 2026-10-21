#!/usr/bin/env python3
"""Summarize complete calibrated multiseed Colab stress run; fail on missing arms."""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ARMS=("adamw","muon_frob","muon_s4","muon_s4_conc",
      "muon_s4_null","muon_s4_null_conc")
KEYS=("valid_ce","copy_acc","bos_real_share","max_column_real_share",
      "source_key_real_share","null_mass","ablation_delta_ce",
      "stressor_delta_ce","high_sink_head_fraction")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    with ZipFile(args.input) as z:
        rows=json.loads(z.read("summary.json"))
        calibrations=json.loads(z.read("calibration.json"))
        cfg=json.loads(z.read("config.json"))
    expected={(s,a) for s in cfg["seeds"] for a in ARMS}
    actual={(r["seed"],r["arm"]) for r in rows}
    assert actual==expected and len(rows)==len(expected),(actual,expected)
    assert len(calibrations)==len(expected)
    assert all(c["delta_bos"]>0.35 and c["stressed_bos_real_share"]>0.5
               for c in calibrations)
    assert all(r["step"]==cfg["steps"] for r in rows)
    mean={}
    for arm in ARMS:
        a=[r for r in rows if r["arm"]==arm]
        mean[arm]={}
        for k in KEYS:
            values=[r[k] for r in a]
            mean[arm][k]={"mean":statistics.mean(values),
                          "sd":statistics.stdev(values),
                          "individual":dict(zip(cfg["seeds"],values))}
    differences={}
    base={(r["seed"],r["arm"]):r for r in rows}
    for arm in ARMS:
        if arm=="muon_s4":continue
        differences[arm]={}
        for k in KEYS:
            pairs=[base[(seed,arm)][k]-base[(seed,"muon_s4")][k]
                   for seed in cfg["seeds"]]
            differences[arm][k]={"mean":statistics.mean(pairs),
                                  "sd":statistics.stdev(pairs),
                                  "individual":dict(zip(cfg["seeds"],pairs))}
    report={
        "metadata":{"seeds":cfg["seeds"],"arms":ARMS,
                    "model_nonnull_parameters":next(r["params"] for r in rows
                       if r["arm"]=="muon_s4"),
                    "steps_per_arm":cfg["steps"],"bos_bias":cfg["bos_bias"],
                    "min_calibration_delta":min(c["delta_bos"] for c in calibrations),
                    "min_calibration_share":min(c["stressed_bos_real_share"]
                                               for c in calibrations),
                    "status":"all_18_completed_all_18_induction_gates_passed"},
        "means_sd":mean,
        "paired_differences_from_muon_s4":differences
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report["metadata"],indent=2))
    for arm in ARMS:
        m=mean[arm]
        print(f"{arm:22s} BOS={m['bos_real_share']['mean']:.3f} "
              f"maxcol={m['max_column_real_share']['mean']:.3f} "
              f"copy={m['copy_acc']['mean']:.3f} "
              f"source={m['source_key_real_share']['mean']:.3f} "
              f"CE={m['valid_ce']['mean']:.3f}")
if __name__=="__main__":
    main()
