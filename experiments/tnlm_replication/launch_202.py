import subprocess,sys,pathlib,json,hashlib
seed=202
corpus=pathlib.Path('/content/gcl_tnlm_fineweb')
m=json.loads((corpus/'corpus_manifest.json').read_text())
assert m['train_sha256']=='953b7aa03a10181db8025c6170acacfb3e242e9978cdf0892b327ab937469552'
assert m['valid_sha256']=='3f158048b74095ac5446cabf8f66596b1aae3b5907c3289e107d20228d25a43a'
assert m['manifest_sha256']=='5233f44b9091a9d4f0af9f5abebfc6bcb64d4da098f7cfddf52bfe91630acbba'
out=pathlib.Path('/content/gcl_fsm_three_seed')
out.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'/content/gcl_rep_train.py','--data',str(corpus),'--out',str(out),
 '--stage','FSM_S4_REPLICATION','--steps','4000','--batch','2','--accum','4','--context','512',
 '--peak-lr','0.0003','--warmup','100','--eval-every','250','--eval-batches','4',
 '--seeds',str(seed),'--arms','fsm_tnlm','fsm_tnlm_s4','fsm_tnlm_s4_col',
 '--lambda-s4','0.15','--lambda-column','0.15','--force-sink-k','16',
 '--retrieval-samples','24','--long-eval','--wandb','offline']
(out/'sources.sha256').write_text(''.join(hashlib.sha256(pathlib.Path(z).read_bytes()).hexdigest()+' '+z+'\n' for z in ('/content/gcl_rep_train.py','/content/retrieval_eval.py')))
f=(out/'run.log').open('w',buffering=1)
p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('REPLICATION_STARTED',p.pid,'seed',seed,'arms',3,'tokens',3*4000*2*4*512,flush=True)
