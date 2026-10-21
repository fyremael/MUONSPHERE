import subprocess,sys,pathlib
out=pathlib.Path('/content/gcl_natural_sink_full')
out.mkdir(exist_ok=True)
f=open(out/'run.log','w',buffering=1)
args=[sys.executable,'/content/gcl_natural_train.py',
 '--data','/content/gcl_natural_sink','--out',str(out),
 '--stage','fineweb_124m_multiseed',
 '--context','512','--microbatch','2','--accum','2',
 '--steps','320','--eval-every','40','--eval-batches','4',
 '--seeds','101','202','303','--arms',
 'muon_frob','muon_s4','muon_s4_conc','adamw','adamw_conc',
 '--lambda-conc','0.03','--wandb-mode','offline',
 '--require-natural-sink','--eval-long-context',
 '--max-training-minutes','30']
proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print("NATURAL_FULL_SPAWNED",proc.pid, "COMMAND",repr(args),flush=True)
