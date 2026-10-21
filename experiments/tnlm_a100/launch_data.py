import subprocess,sys,pathlib
out=pathlib.Path('/content/gcl_tnlm_fineweb');out.mkdir(parents=True,exist_ok=True)
f=open(out/'prepare.log','w',buffering=1)
cmd=[sys.executable,'/content/gcl_tnlm_prepare.py',
'--out',str(out),'--train-tokens','60000000','--valid-tokens','4000000']
p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print("PREP_SPAWNED",p.pid,flush=True)
