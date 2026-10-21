"""Matched from-scratch attention regularization arms on same text and RNG."""
import subprocess,sys,pathlib
out=pathlib.Path('/content/gcl_s4_scratch_comparison')
out.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'/content/gcl_s4_scratch_train.py',
 '--data','/content/gcl_s4_scratch_corpus',
 '--out',str(out),'--stage','random_init_s4_vs_s4plus',
 '--steps','4000','--batch','2','--accum','4','--context','512',
 '--peak-lr','0.0003','--warmup','100',
 '--eval-every','250','--eval-batches','4',
 '--seeds','101','--arms','s4','s4_plus_penalty',
 '--lambda-s4','0.15','--lambda-column','0.15',
 '--long-eval','--wandb','offline']
f=open(out/'run.log','w',buffering=1)
p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('S4_COMPARISON_DISPATCHED',p.pid,repr(cmd),flush=True)
