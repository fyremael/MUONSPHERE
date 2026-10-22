import subprocess,pathlib,sys
p=pathlib.Path('/content/gcl_tnlm_fineweb');p.mkdir(parents=True,exist_ok=True)
f=(p/'prepare.log').open('w',buffering=1)
c=[sys.executable,'/content/gcl_rep_prepare.py','--out',str(p),'--train-tokens','60000000','--valid-tokens','4000000']
z=subprocess.Popen(c,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('PREPARATION_START',z.pid,flush=True)
