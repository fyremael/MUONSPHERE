"""Package exact A100 run outputs and offline W&B for private CHAD custody."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,hashlib
root=Path('/content')
ind=root/'gcl_tnlm_induction'
status=json.loads((ind/'run_status.json').read_text())
assert status['complete'] and status['count']==2
if (root/'gcl_tnlm_s4_ablation'/'run_status.json').exists():
    abstatus=json.loads((root/'gcl_tnlm_s4_ablation'/'run_status.json').read_text())
    assert abstatus['complete'] and abstatus['count']==2,abstatus
run_dirs=[ind]
if (root/'gcl_tnlm_s4_ablation'/'run_status.json').exists():
    run_dirs.append(root/'gcl_tnlm_s4_ablation')
payload=[]
for folder in run_dirs:
    for name in ('config.json','metrics.jsonl','initial.jsonl','summary.json','run_status.json','run.log'):
        p=folder/name
        if p.is_file():payload.append(p)
    payload.extend(folder.rglob('*.wandb'))
p=root/'gcl_tnlm_fineweb'/'corpus_manifest.json'
if p.is_file():payload.append(p)
p=ind/'dispatch_decision.json'
if p.is_file():payload.append(p)
archive=root/'gcl_tnlm_a100_private_receipts.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for f in payload:z.write(f,str(f.relative_to(root)))
print('PRIVATE_ARCHIVE',json.dumps({
 'file_count':len(payload),'byte_count':archive.stat().st_size,
 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'contains_offline_wandb':any(str(f).endswith('.wandb') for f in payload),
 'complete_runs':[d.name for d in run_dirs]}),flush=True)
