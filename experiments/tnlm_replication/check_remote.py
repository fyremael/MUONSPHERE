from pathlib import Path
import json,time,subprocess
p=Path('/content/gcl_tnlm_fineweb')
print('TIME',time.time(),flush=True)
for f in ('corpus_manifest.json','train_tokens.uint16','valid_tokens.uint16','prepare.log'):
 x=p/f
 print(f,x.exists(),x.stat().st_size if x.exists() else None,flush=True)
if (p/'corpus_manifest.json').exists():
 m=json.loads((p/'corpus_manifest.json').read_text())
 print('HASHES',m['train_sha256'],m['valid_sha256'],m['manifest_sha256'],flush=True)
if (p/'prepare.log').exists(): print('PREP',*(p/'prepare.log').read_text(errors='replace').splitlines()[-3:],sep='\n',flush=True)
