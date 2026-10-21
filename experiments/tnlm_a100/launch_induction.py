"""Detached training; keep long Colab execution independent of CLI connection."""
import pathlib,subprocess,sys,json
manifest=json.load(open('/content/gcl_tnlm_fineweb/corpus_manifest.json'))
assert manifest['train_tokens']==60000000 and manifest['valid_tokens']==4000000
assert manifest['manifest_sha256']=='5233f44b9091a9d4f0af9f5abebfc6bcb64d4da098f7cfddf52bfe91630acbba'
out=pathlib.Path('/content/gcl_tnlm_induction')
out.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'/content/gcl_tnlm_train.py',
 '--data','/content/gcl_tnlm_fineweb','--out',str(out),
 '--stage','TNLM_induction_pilot','--context','512',
 '--batch','2','--accum','4','--steps','4000','--warmup','100',
 '--peak-lr','0.0003','--eval-every','250','--eval-batches','4',
 '--seeds','101','--arms','tnlm','fsm_tnlm',
 '--force-sink-k','16','--long-eval','--wandb','offline']
with open(out/'run.log','w',buffering=1) as log:
    task=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True)
print('INDUCTION_DISPATCHED',json.dumps({'pid':task.pid,'cmd':cmd}),flush=True)
