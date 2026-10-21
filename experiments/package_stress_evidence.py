#!/usr/bin/env python3
"""Reproduce the public, redacted evidence bundle from the completed T4 run."""
import argparse, hashlib, json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PREFIX="attention_sink_s4_stress_20261021_"
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"docs/data"
SOURCE_SHA="abf75a07cba8873ff038b12571760d354fcf95b97ccae52b16e2c5bab6a4092b"

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-archive",type=Path,required=True)
    p.add_argument("--control-log",type=Path,required=True)
    args=p.parse_args()
    with ZipFile(args.run_archive) as z:
        for name in ("config.json","calibration.json","metrics.jsonl",
                     "summary.json","summary.csv","positive_control.json"):
            raw=z.read(name)
            if name.endswith(".csv"):
                raw=raw.replace(b"\r\n",b"\n")
            (DATA/(PREFIX+name)).write_bytes(raw)
    matches=[l.removeprefix("HOSTILE_METRIC_CONTROLS ") for l in
             args.control_log.read_text().splitlines()
             if l.startswith("HOSTILE_METRIC_CONTROLS ")]
    assert len(matches)==1
    (DATA/(PREFIX+"hostile_controls.json")).write_text(
        json.dumps(json.loads(matches[0]),indent=2,sort_keys=True)+"\n")
    source=(ROOT/"experiments/attention_sink_s4_stress_colab.py").read_bytes()
    exact=source+b"\n"
    assert hash_bytes(exact)==SOURCE_SHA, hash_bytes(exact)
    files={}
    selected=[
        "experiments/attention_sink_s4.py",
        "experiments/attention_sink_s4_stress.py",
        "experiments/attention_sink_s4_stress_colab.py",
        "experiments/build_stress_colab.py",
        "experiments/test_attention_sink_stress_adversaries.py",
        "experiments/test_attention_sink_stress_adversaries_colab.py",
        "experiments/build_stress_controls_colab.py",
        "experiments/summarize_stress.py",
        "docs/attention_sink_s4_stress_protocol.md",
        "docs/attention_sink_s4_stress_results_20261021.md"]
    selected += [str(x.relative_to(ROOT)) for x in sorted(DATA.glob(PREFIX+"*"))
                 if x.suffix in (".json",".jsonl",".csv") and
                 x.name != PREFIX+"manifest.json"]
    for name in selected:
        files[name]=(ROOT/name).read_bytes()
    files["exact_colab_execution_source.py"]=exact
    config=json.loads((DATA/(PREFIX+"config.json")).read_text())
    rows=json.loads((DATA/(PREFIX+"summary.json")).read_text())
    checks=json.loads((DATA/(PREFIX+"calibration.json")).read_text())
    assert len(rows)==len(checks)==18
    assert all(x["delta_bos"]>0.35 for x in checks)
    manifest={
        "record_type":"GCL_ATTENTION_SINK_STRESS_RECEIPT/1",
        "run_archive_sha256":hash_bytes(args.run_archive.read_bytes()),
        "completed_arm_seed_runs":18,
        "induction_gates_passed":18,
        "seeds":config["seeds"],
        "arms":config["arms"],
        "colab_session":"gcl-stress-big-1021",
        "completion_exit_code":0,
        "colab_run_source_sha256":SOURCE_SHA,
        "files_sha256":{n:hash_bytes(b) for n,b in sorted(files.items())},
        "wandb":"Recorded offline; unredacted W&B files excluded from public bundle"}
    manifest_bytes=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    files["manifest.json"]=manifest_bytes
    (DATA/(PREFIX+"manifest.json")).write_bytes(manifest_bytes)
    bundle=DATA/(PREFIX+"bundle.zip")
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        for name,content in sorted(files.items()):
            z.writestr(name,content)
    with ZipFile(bundle) as z:
        for name,expected_sha in manifest["files_sha256"].items():
            assert hash_bytes(z.read(name))==expected_sha
    print("BUNDLE_VERIFIED",bundle,len(files),
          "SHA256",hash_bytes(bundle.read_bytes()))
if __name__=="__main__":
    main()
