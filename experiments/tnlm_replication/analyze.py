#!/usr/bin/env python3
"""Strict three-seed A100 FSM/S4 replication with paired retrieval evidence."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

ARMS=("fsm_tnlm","fsm_tnlm_s4","fsm_tnlm_s4_col")
SEEDS=(101,202,303)
CORE=("bos_share","max_column_share","strongest_layer_sink_fraction",
      "sink_head_fraction","valid_ce","long_valid_ce","attention_entropy")
RETR=("natural_copy_nll","natural_copy_ablate_nll","natural_copy_lift_nats",
      "natural_copy_top1","natural_copy_ablate_top1","passkey_fourway_correct",
      "passkey_ablation_true_logit_lift")
T_DF2=4.302652729911275

def loadj(path):
    return json.loads(path.read_text())
def jl(path):
    return [json.loads(t) for t in path.read_text().splitlines() if t]
def gate(rows):
    seq=sorted([r for r in rows if r["step"]>1],key=lambda r:r["step"])
    for j in range(12):
        for i in range(len(seq)-2):
            z=seq[i:i+3]
            if all(q["sink_fraction_by_layer"][j]>=.25 for q in z):
                return dict(passed=True,layer=j+1,
                    steps=[q["step"] for q in z],
                    fractions=[q["sink_fraction_by_layer"][j] for q in z])
    return dict(passed=False,max_layer_head_fraction=max(
        max(r["sink_fraction_by_layer"]) for r in seq))

def agg(values):
    mean=statistics.mean(values)
    std=statistics.stdev(values)
    return dict(mean=mean,sd=std,CI95=[mean-T_DF2*std/math.sqrt(3),
                                     mean+T_DF2*std/math.sqrt(3)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipts",default=str(Path(__file__).parent/"receipts"))
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    root=Path(args.receipts)
    finals={}
    per_seed={}
    fingerprints={}
    shared_manifest=None
    for seed in SEEDS:
        p=root/f"seed{seed}"
        needed=("config.json","summary.json","run_status.json",
                "initial.jsonl","metrics.jsonl","retrieval.jsonl","sources.sha256")
        for n in needed:
            assert (p/n).exists(),f"Missing {p/n}; refuse completion"
        config=loadj(p/"config.json")
        status=loadj(p/"run_status.json")
        summary=loadj(p/"summary.json")
        init=jl(p/"initial.jsonl")
        metrics=jl(p/"metrics.jsonl")
        retrieval=jl(p/"retrieval.jsonl")
        assert config["steps"]==4000 and config["seeds"]==[seed] and config["arms"]==list(ARMS)
        assert config["batch"]==2 and config["accum"]==4 and config["context"]==512
        assert config["peak_lr"]==0.0003 and config["warmup"]==100
        assert config["lambda_s4"]==0.15 and config["lambda_column"]==0.15
        assert config["force_sink_k"]==16 and config["pretraining_checkpoint"] is None
        assert config["retrieval_samples"]==24 and config["long_eval"]
        assert status["complete"] and status["count"]==status["expected"]==3
        manifest=config["corpus"]
        if shared_manifest is None:shared_manifest=manifest
        for key in ("manifest_sha256","train_sha256","valid_sha256"):
            assert manifest[key]==shared_manifest[key],(seed,key)
        assert len(summary)==len(init)==len(retrieval)==3
        assert {s["arm"] for s in summary}==set(ARMS)
        assert {s["arm"] for s in init}==set(ARMS)
        assert {s["arm"] for s in retrieval}==set(ARMS)
        fingerprints[seed]={z["model_sha256_sample"] for z in init}
        assert len(fingerprints[seed])==1,(seed,fingerprints[seed])
        finals[seed]={}
        per_seed[seed]={}
        for arm in ARMS:
            entries=[s for s in summary if s["arm"]==arm]
            assert len(entries)==1 and entries[0]["completed"] and entries[0]["step"]==4000
            last=entries[0]["last"]
            assert last["step"]==4000 and last["seed"]==seed and last["arm"]==arm
            trajectory=[r for r in metrics if r["arm"]==arm]
            assert trajectory and trajectory[-1]["step"]==4000
            ret=[r for r in retrieval if r["arm"]==arm]
            assert len(ret)==1 and ret[0]["step"]==4000
            assert last["retrieval"]==ret[0]["retrieval"]
            for length in ("512","1024"):
                assert last["retrieval"][length]["n"]==24
                assert all(math.isfinite(last["retrieval"][length][key]) for key in RET)
            finals[seed][arm]=last
            per_seed[seed][arm]=gate(trajectory)
    assert len({next(iter(fingerprints[s])) for s in SEEDS})==3
    stats={}
    for arm in ARMS:
        stats[arm]={key:agg([finals[seed][arm][key] for seed in SEEDS]) for key in CORE}
        stats[arm]["retrieval"]={
         length:{key:agg([finals[s][arm]["retrieval"][length][key] for s in SEEDS])
                   for key in RET}
         for length in ("512","1024")}
    paired={}
    for arm in ARMS[1:]:
        paired[arm]={key:agg([finals[s][arm][key]-
                            finals[s][ARMS[0]][key] for s in SEEDS]) for key in CORE}
        paired[arm]["retrieval"]={
          length:{key:agg([finals[s][arm]["retrieval"][length][key]-
                              finals[s][ARMS[0]]["retrieval"][length][key]
                      for s in SEEDS]) for key in RET}
          for length in ("512","1024")}
    result=dict(record_type="GCL_FSM_S4_THREE_SEED_RETRIEVAL/1",
       status="COMPLETE",model="GPT2-124M-random-init",
       seeds=SEEDS,arms=ARMS,
       hardware="A100-40GB",dataset_sha256=shared_manifest["manifest_sha256"],
       train_tokens_per_arm=4000*2*4*512,
       total_processed_positions=3*3*4000*2*4*512,
       gates=per_seed,initial_sample_fingerprints={
         str(k):next(iter(v)) for k,v in fingerprints.items()},
       final_metrics=finals,stats=stats,paired_differences=paired,
       source="Three independent seeds; no injected attention-logit bias; "
              "Force-sink Masking is structural induction, not spontaneous emergence.",
       retrieval_limitations="Zero-shot passage-copy and 4-choice passkey probes; "
              "not supervised retrieval training, and not proof of useful retrieval "
              "unless retrieval-ablation lift and accuracy exceed chance.")
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("COMPLETE",result["status"],"seeds",result["seeds"])
    for arm in ARMS:
        z=stats[arm]
        print(arm,"BOS",round(z["bos_share"]["mean"],4),
              "peakhead",round(z["strongest_layer_sink_fraction"]["mean"],4),
              "CE",round(z["valid_ce"]["mean"],4))
        for length in ("512","1024"):
            rz=z["retrieval"][length]
            print("RETR",length,"copy_lift",round(rz["natural_copy_lift_nats"]["mean"],5),
                  "passkey",round(rz["passkey_fourway_correct"]["mean"],4))
if __name__=="__main__":main()

