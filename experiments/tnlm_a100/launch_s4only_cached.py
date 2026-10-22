import pathlib,subprocess,sys,json,hashlib
cache=pathlib.Path('/content/gcl_tnlm_fineweb')
m=json.loads((cache/'corpus_manifest.json').read_text())
assert m['train_sha256']=='953b7aa03a10181db8025c6170acacfb3e242e9978cdf0892b327ab937469552'
assert m['valid_sha256']=='3f158048b74095ac5446cabf8f66596b1aae3b5907c3289e107d20228d25a43a'
for name in ('train','valid'):
 f=cache/(name+'_tokens.uint16')
 assert hashlib.sha256(f.read_bytes()).hexdigest()==m[name+'_sha256']
dest=pathlib.Path('/content/gcl_fsm_s4only_direct')
dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'run_status.json').exists()
cmd=[sys.executable,'/content/gcl_tnlm_train.py',
 '--data',str(cache),'--out',str(dest),
 '--stage','FSM_S4_only_direct_verified_corpus',
 '--context','512','--batch','2','--accum','4',
 '--steps','4000','--warmup','100','--peak-lr','0.0003',
 '--eval-every','250','--eval-batches','4',
 '--seeds','101','--arms','fsm_tnlm_s4','--force-sink-k','16',
 '--lambda-s4','0.15','--lambda-column','0.15',
 '--long-eval','--wandb','offline']
f=(dest/'run.log').open('w',buffering=1)
p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('S4_ONLY_DIRECT_STARTED',p.pid,'args',cmd,flush=True)
