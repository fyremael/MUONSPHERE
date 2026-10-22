import subprocess,sys,pathlib
out=pathlib.Path('/content/gcl_s4_scratch_pilot');out.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'/content/gcl_s4_scratch_train.py',
     '--data','/content/gcl_s4_scratch_corpus',
     '--out',str(out),'--stage','pilot',
     '--steps','320','--batch','2','--accum','2','--context','512',
     '--warmup','40','--eval-every','40','--eval-batches','4',
     '--seeds','101','--arms','adamw',
     '--wandb','disabled','--long-eval']
f=open(out/'run.log','w',buffering=1)
proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('PILOT_DISPATCHED',proc.pid,cmd,flush=True)
