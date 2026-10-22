#!/usr/bin/env python3
"""Create verified public ZIP of completed multi-session A100 scratch S4 study."""
import json,hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
DEST=REPO/"docs/data"

def sha(data): return hashlib.sha256(data).hexdigest()
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    summary=json.loads((DEST/"scratch_s4_20261021_analysis.json").read_text())
    assert summary["status"]=="THREE_ARMS_AT_4000_STEPS"
    sources=[
        "experiments/fromscratch_a100/README.md",
        "experiments/fromscratch_a100/prepare_fineweb.py",
        "experiments/fromscratch_a100/train_scratch.py",
        "experiments/fromscratch_a100/reconcile_runs.py",
        "experiments/fromscratch_a100/test_s4_math.py",
        "experiments/fromscratch_a100/package_reconciled.py",
        "docs/attention_sink_s4_fromscratch_20261021_report.md",
    ]
    inputs={}
    for item in sources:
        inputs["source/"+item]=(REPO/item).read_bytes()
    for key,folder in (("baseline",HERE/"evidence/baseline"),
                       ("combined_replay",HERE/"evidence/final")):
        for name in ("config.json","metrics.jsonl","summary.json",
                     "initial.jsonl","run_status.json"):
            inputs[f"runs/{key}/{name}"]=(folder/name).read_bytes()
    oldrows=[json.loads(x) for x in (HERE/"compare-live.jsonl").read_text().splitlines()]
    s4=[x for x in oldrows if x["arm"]=="s4"]
    partial=[x for x in oldrows if x["arm"]=="s4_plus_penalty"]
    assert s4[-1]["step"]==4000 and len(s4)==17
    assert partial[-1]["step"]<4000
    inputs["runs/s4_completed/metrics.jsonl"]=(
          "\n".join(json.dumps(row,sort_keys=True) for row in s4)+"\n").encode()
    inputs["runs/failed_combined_attempt/incomplete_metrics.jsonl"]=(
          "\n".join(json.dumps(row,sort_keys=True) for row in partial)+"\n").encode()
    log=(HERE/"compare-live.log").read_text().splitlines()
    for typ,name in (("RUN_CONFIG ","config.json"),
                     ("INITIAL ","initial.jsonl")):
        payload=[json.loads(x[len(typ):]) for x in log if x.startswith(typ)]
        assert len(payload)==(1 if name=="config.json" else 2)
        inputs["runs/s4_completed/"+name]=(
          json.dumps(payload[0],indent=2,sort_keys=True)+"\n" if
          name=="config.json" else
          "".join(json.dumps(row,sort_keys=True)+"\n" for row in payload)).encode()
    inputs["analysis.json"]=(DEST/"scratch_s4_20261021_analysis.json").read_bytes()
    reconstructed=(REPO/"experiments/fromscratch_a100/train_scratch.py").read_bytes()+b"\n"
    inputs["source/executed_uploaded_runner.py"]=reconstructed
    manifest={
      "record_type":"GCL_A100_TRUE_S4_FROM_SCRATCH_EVIDENCE/1",
      "status":"three arms completed; two A100 sessions",
      "arms":summary["arms"],
      "optimizer":"AdamW only",
      "seed":101,
      "steps_each":4000,
      "tokens_per_arm":summary["training_token_positions_per_arm"],
      "original_comparison_interrupted":True,
      "first_s4_arm_completed":True,
      "combined_arm_replayed_to_completion":True,
      "interrupted_partial_combined_attempt_excluded":True,
      "training_corpus_manifest_sha256":summary["dataset_manifest_sha256"],
      "reconstructed_uploaded_runner_sha256":sha(reconstructed),
      "filenames_sha256":{name:sha(data) for name,data in sorted(inputs.items())},
      "raw_text_or_checkpoint_included":False,
      "interpretation":"no pronounced baseline sink; prevention not demonstrated"
    }
    inputs["manifest.json"]=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    target=DEST/"scratch_s4_20261021_bundle.zip"
    with ZipFile(target,"w",ZIP_DEFLATED) as zipf:
        for path,data in sorted(inputs.items()):
            zipf.writestr(path,data)
    with ZipFile(target) as zipf:
        for path,h in manifest["filenames_sha256"].items():
            assert sha(zipf.read(path))==h,path
    (DEST/"scratch_s4_20261021_manifest.json").write_bytes(inputs["manifest.json"])
    print("VALIDATED",target.stat().st_size,"bytes",sha(target.read_bytes()),
          "files",len(inputs),flush=True)
if __name__=="__main__":main()

