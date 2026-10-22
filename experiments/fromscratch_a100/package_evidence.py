#!/usr/bin/env python3
"""Verify and package the Colab scratch GPT2 S4 three-arm result.

Public ZIP excludes private W&B binary/runtime IDs and full HF corpus.
Every source file and observation is hashed; no invented run metrics.
"""
import argparse,hashlib,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
EXPECTED=("adamw","s4","s4_plus_penalty")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",type=Path,default=HERE/"evidence/baseline")
    ap.add_argument("--compare",type=Path,default=HERE/"evidence/comparison")
    ap.add_argument("--out",type=Path,default=REPO/"docs/data")
    opts=ap.parse_args()
    opts.out.mkdir(parents=True,exist_ok=True)
    groups={"baseline":opts.base,"comparison":opts.compare}
    files={}
    summaries={}
    initials={}
    config={}
    for group,folder in groups.items():
        for name in ("config.json","metrics.jsonl","summary.json",
                     "initial.jsonl","run_status.json"):
            path=folder/name
            assert path.exists(),path
            files[f"observations/{group}/{name}"]=path.read_bytes()
        config[group]=json.loads((folder/"config.json").read_text())
        status=json.loads((folder/"run_status.json").read_text())
        assert status["complete"] and status["count"]==status["expected"],status
        summaries[group]=json.loads((folder/"summary.json").read_text())
        initials[group]=[json.loads(x) for x in
             (folder/"initial.jsonl").read_text().splitlines()]
    assert len(summaries["baseline"])==1
    assert len(summaries["comparison"])==2
    arms=tuple(x["arm"] for x in
               summaries["baseline"]+summaries["comparison"])
    assert arms==EXPECTED,arms
    basecfg=config["baseline"]
    compcfg=config["comparison"]
    for attr in ("steps","batch","accum","context","peak_lr",
                 "warmup","eval_every","eval_batches","seeds",
                 "model","pretraining_checkpoint"):
        assert basecfg[attr]==compcfg[attr],attr
    for key in ("train_sha256","valid_sha256","manifest_sha256"):
        assert basecfg["corpus"][key]==compcfg["corpus"][key],key
    assert basecfg["pretraining_checkpoint"] is None
    assert basecfg["seeds"]==[101]
    assert basecfg["steps"]==4000
    fingerprints=[record["model_sha256_sample"] for group in groups
         for record in initials[group]]
    assert len(fingerprints)==3 and len(set(fingerprints))==1, fingerprints
    for group in groups:
        for record in summaries[group]:
            assert record["completed"] and record["step"]==4000
            assert record["last"]["step"]==4000
    analysis=json.loads((opts.out/"scratch_s4_20261021_analysis.json").read_text())
    assert analysis["record_type"]=="GCL_S4_SCRATCH_A100_RESULT/1"
    assert analysis["status"]=="COMPLETE"
    assert set(analysis["final"])==set(EXPECTED)
    files["analysis.json"]=(opts.out/"scratch_s4_20261021_analysis.json").read_bytes()
    local_sources=[
        "experiments/fromscratch_a100/train_scratch.py",
        "experiments/fromscratch_a100/prepare_fineweb.py",
        "experiments/fromscratch_a100/analyze.py",
        "experiments/fromscratch_a100/test_s4_math.py",
        "experiments/fromscratch_a100/README.md",
        "docs/attention_sink_s4_fromscratch_20261021_report.md"
    ]
    for path in local_sources:
        files[f"source/{path}"]=(REPO/path).read_bytes()
    executed=(HERE/"train_scratch.py").read_bytes()+b"\n"
    # Verified to be exactly the 2026-10-21 uploaded runner, aside from
    # the normalized trailing newline in the GitHub source copy.
    files["source/executed_colab_runner.py"]=executed
    manifest={
        "record_type":"GCL_S4_ATTENTION_FROM_SCRATCH_A100_EVIDENCE/1",
        "status":"complete",
        "induced_attention_bias":False,
        "pretrained_checkpoint":None,
        "optimizer":"AdamW",
        "gpu":basecfg["gpu"],
        "seeds":basecfg["seeds"],
        "arms":EXPECTED,
        "completed_arms":3,
        "steps_each":basecfg["steps"],
        "tokens_each":basecfg["steps"]*basecfg["batch"]*
                      basecfg["accum"]*basecfg["context"],
        "corpus":basecfg["corpus"],
        "shared_initial_sample_sha256":fingerprints[0],
        "uploaded_runner_reconstructed_sha256":sha(executed),
        "files_sha256":{k:sha(v) for k,v in sorted(files.items())},
        "claim":"No pronounced sink emerged in unregularized baseline;"
                " any regularizer results address concentration, not proven prevention",
        "private_WandB":"not published; source run's offline telemetry remains on Colab only unless exported"
    }
    files["manifest.json"]=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    target=opts.out/"scratch_s4_20261021_bundle.zip"
    with ZipFile(target,"w",ZIP_DEFLATED) as z:
        for k,v in sorted(files.items()):
            z.writestr(k,v)
    with ZipFile(target) as z:
        for k,v in manifest["files_sha256"].items():
            assert sha(z.read(k))==v,k
    (opts.out/"scratch_s4_20261021_manifest.json").write_bytes(files["manifest.json"])
    print("BUNDLE_VERIFIED",str(target),"bytes",target.stat().st_size,
          "sha256",sha(target.read_bytes()),"files",len(files))
if __name__=="__main__":
    main()
