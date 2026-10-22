#!/usr/bin/env python3
"""Private Colab receipt custody: excludes raw FineWeb text from archive."""
import hashlib,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path('/content')
sources=(
    ROOT/'gcl_fsm_three_seed',
    ROOT/'gcl_fsm_seed101')
active=[d for d in sources if d.exists()]
assert active
all_files=[]
for d in active:
    status=json.loads((d/'run_status.json').read_text())
    assert status['complete'] and status['count']==status['expected']==3,status
    summary=json.loads((d/'summary.json').read_text())
    assert len(summary)==3 and all(x['completed'] and x['step']==4000 for x in summary)
    for name in ('config.json','metrics.jsonl','initial.jsonl',
                 'retrieval.jsonl','summary.json','run_status.json',
                 'run.log','sources.sha256'):
        p=d/name
        assert p.exists(),p
        all_files.append(p)
    all_files.extend(d.rglob('*.wandb'))
manifest=ROOT/'gcl_tnlm_fineweb'/'corpus_manifest.json'
assert manifest.exists()
all_files.append(manifest)
out=ROOT/'gcl_three_seed_private_receipts.zip'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for p in all_files:
        z.write(p,p.relative_to(ROOT))
with ZipFile(out) as z:
    assert len(z.namelist())==len(all_files)
print("PRIVATE_RECEIPTS_VERIFIED",json.dumps({
    "folders":[x.name for x in active],"bytes":out.stat().st_size,
    "sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
    "files":len(all_files),
    "wandb_captured":sum(p.suffix=='.wandb' for p in all_files)
}),flush=True)
