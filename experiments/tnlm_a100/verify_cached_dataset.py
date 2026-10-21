from pathlib import Path
import json,hashlib
p=Path('/content/gcl_tnlm_fineweb')
m=json.loads((p/'corpus_manifest.json').read_text())
for name,expect in [('train','953b7aa03a10181db8025c6170acacfb3e242e9978cdf0892b327ab937469552'),('valid','3f158048b74095ac5446cabf8f66596b1aae3b5907c3289e107d20228d25a43a')]:
 f=p/(name+'_tokens.uint16')
 print('CORPUS',name,'bytes',f.stat().st_size,'manifest_match',m[name+'_sha256']==expect,'file_match',hashlib.sha256(f.read_bytes()).hexdigest()==expect,flush=True)
print('CONTROLLER_STATUS',json.loads(Path('/content/gcl_fsm_s4_only/controller_status.json').read_text()).get('status'))
