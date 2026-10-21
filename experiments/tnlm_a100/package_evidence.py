#!/usr/bin/env python3
"""Build public provenance package for A100 TNLM/FSM induction vs S4."""
import json,hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
DEST=REPO/"docs/data"

def sha(x):return hashlib.sha256(x).hexdigest()
def main():
    report=json.loads((DEST/"tnlm_fsm_20261021_analysis.json").read_text())
    assert report["status"]=="COMPLETE"
    assert report["gate"]["fsm_tnlm"]["passed"]
    contents={}
    paths=[
        "experiments/tnlm_a100/README.md",
        "experiments/tnlm_a100/train_tnlm.py",
        "experiments/tnlm_a100/test_mask.py",
        "experiments/tnlm_a100/prepare_fineweb.py",
        "experiments/tnlm_a100/analyze_induction.py",
        "experiments/tnlm_a100/reconcile.py",
        "experiments/tnlm_a100/package_evidence.py",
        "experiments/tnlm_a100/launch_s4only_cached.py",
        "experiments/tnlm_a100/finalize_direct_s4.py",
        "experiments/tnlm_a100/archive_direct.py",
        "experiments/tnlm_a100/report.py",
        "experiments/tnlm_a100/watch_s4only_direct.sh",
        "docs/attention_sink_tnlm_fsm_20261021_report.md",
    ]
    for name in paths:
        contents["source/"+name]=(REPO/name).read_bytes()
    evidence=[
      ("induction",["induction-metrics.jsonl","induction-run.log"]),
      ("replay",["replay-config.json","replay-initial.jsonl",
                 "replay-metrics.jsonl","replay-summary.json","replay-run_status.json",
                 "replay-corpus-manifest.json"]),
      ("s4col",["s4col-config.json","s4col-initial.jsonl",
                 "s4col-metrics.jsonl","s4col-summary.json",
                 "s4col-run_status.json","s4col-corpus-manifest.json"]),
      ("optional_s4only",["s4only-config.json","s4only-initial.jsonl",
                 "s4only-metrics.jsonl","s4only-summary.json","s4only-run_status.json",
                 "s4only-controller-status.json", "s4only-failed-controller-status.json"])]
    for group,files in evidence:
        for name in files:
            path=HERE/name
            if not path.is_file():
                assert group=="optional_s4only",(group,name)
                continue
            contents[f"runs/{group}/{name}"]=path.read_bytes()
    contents["analysis.json"]=(DEST/"tnlm_fsm_20261021_analysis.json").read_bytes()
    manifest={
      "record_type":"GCL_TNLM_FSM_ATTENTION_SINK_A100_EVIDENCE/1",
      "training_induction":"TNLM with fixed-channel FSM (structural)",
      "natural_language_corpus":"FineWeb-Edu sample-10BT",
      "model":"GPT2-Small 124M random initialization",
      "induced_from_attention_logit_bias":False,
      "run_status":report["status"],
      "qualified_sink_gate":report["gate"]["fsm_tnlm"],
      "original_fsm_trace_interrupted":True,
      "original_interrupted_fsm_step":report["original_interrupted_fsm_step"],
      "interrupted_data_excluded_from_matched_final":True,
      "corrected_complete_fsm_included":True,
      "trained_parameter_fingerprint":report["model_initial_sample_sha256"],
      "document_manifest_sha":report["corpus_document_hash"],
      "public_files_sha256":{n:sha(b) for n,b in sorted(contents.items())},
      "private_WandB":"not republished to GitHub"}
    contents["manifest.json"]=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    output=DEST/"tnlm_fsm_20261021_bundle.zip"
    with ZipFile(output,"w",ZIP_DEFLATED) as z:
        for name,data in sorted(contents.items()):z.writestr(name,data)
    with ZipFile(output) as z:
        for name,expected in manifest["public_files_sha256"].items():
            assert sha(z.read(name))==expected,name
    (DEST/"tnlm_fsm_20261021_manifest.json").write_bytes(contents["manifest.json"])
    print("EVIDENCE_PACKAGE_VALID",output.stat().st_size,
          sha(output.read_bytes()),"FILES",len(contents))
if __name__=="__main__":main()
