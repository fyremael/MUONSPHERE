#!/usr/bin/env python3
"""Build auditable public A100 evidence bundle from retrieved private Colab run."""
import argparse,hashlib,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

SUFFIX="natural_a100_20261021"
SOURCE_SHA="ee0831998326880ee4063f6a88776da0c2b262b19b607b5224316faf7accbb93"

def sha(data):return hashlib.sha256(data).hexdigest()
def make(repo,private_zip):
    repo=Path(repo)
    dest=repo/"docs/data"
    dest.mkdir(parents=True,exist_ok=True)
    with ZipFile(private_zip) as z:
        names=("config.json","initial_metrics.jsonl","metrics.jsonl",
               "summary.json","run_status.json","corpus_manifest.json")
        payload={f"results/{name}":z.read(name) for name in names}
    cfg=json.loads(payload["results/config.json"])
    done=json.loads(payload["results/run_status.json"])
    assert done=={"status":"complete","completed":15,"expected":15},done
    assert cfg["seeds"]==[101,202,303] and cfg["steps"]==320,cfg
    sources=[
       "experiments/natural_a100/README.md",
       "experiments/natural_a100/train_natural.py",
       "experiments/natural_a100/prepare_fineweb.py",
       "experiments/natural_a100/analyze_results.py",
       "docs/attention_sink_natural_a100_20261021_report.md"]
    for name in sources:
        payload[name]=(repo/name).read_bytes()
    executed=(repo/"experiments/natural_a100/train_natural.py").read_bytes()+b"\n"
    assert sha(executed)==SOURCE_SHA
    payload["exact_colab_training_source.py"]=executed
    analysis=repo/"docs/data"/f"{SUFFIX}_analysis.json"
    assert analysis.exists()
    payload["results/analysis.json"]=analysis.read_bytes()
    token_info=json.loads(payload["results/corpus_manifest.json"])
    assert token_info["train_tokens"]==12000000 and token_info["valid_tokens"]==1000000
    manifest={
        "record_type":"GCL_NATURAL_ATTENTION_SINK_A100_RECEIPT/1",
        "status":"COMPLETE",
        "train_pretrained_not_random_init":True,
        "n_runs":15,
        "seed_count":3,
        "arms":cfg["arms"],
        "colab_gpu":cfg["device_name"],
        "source_checkpoint":cfg["pretraining_checkpoint"],
        "executed_script_sha256":SOURCE_SHA,
        "train_corpus_sha256":token_info["train_sha256"],
        "valid_corpus_sha256":token_info["valid_sha256"],
        "files_sha256":{name:sha(data) for name,data in sorted(payload.items())},
        "notes":"W&B offline binary retained in private CHAD archive, not in public GitHub bundle"}
    payload["manifest.json"]=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    out=dest/f"{SUFFIX}_bundle.zip"
    with ZipFile(out,"w",ZIP_DEFLATED) as z:
        for name,data in sorted(payload.items()):z.writestr(name,data)
    with ZipFile(out) as z:
        for name,expected in manifest["files_sha256"].items():
            assert sha(z.read(name))==expected,name
    (dest/f"{SUFFIX}_manifest.json").write_bytes(payload["manifest.json"])
    print("VERIFIED_BUNDLE",out,"bytes",out.stat().st_size,
          "sha256",sha(out.read_bytes()),"files",len(payload))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo",default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--private-zip",required=True)
    args=p.parse_args()
    make(args.repo,args.private_zip)
