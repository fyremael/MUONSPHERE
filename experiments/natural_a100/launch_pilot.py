import subprocess,sys,pathlib
out=pathlib.Path('/content/gcl_natural_sink_pilot')
out.mkdir(exist_ok=True)
file=open(out/'run.log','w',buffering=1)
cmd=[sys.executable,'/content/gcl_natural_train.py','--data','/content/gcl_natural_sink',
 '--out',str(out),'--stage','pilot',
 '--context','256','--microbatch','1','--accum','1',
 '--steps','8','--eval-every','4','--eval-batches','2',
 '--seeds','101','--arms','muon_s4','muon_s4_conc',
 '--wandb-mode','disabled']
proc=subprocess.Popen(cmd,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
print("PILOT_SPAWNED",proc.pid,flush=True)
