import subprocess,sys,os,pathlib
folder=pathlib.Path('/content/gcl_natural_sink')
folder.mkdir(parents=True,exist_ok=True)
f=open(folder/'prepare.log','w',buffering=1)
p=subprocess.Popen([sys.executable,'/content/gcl_prepare_fineweb.py',
    '--train-tokens','12000000','--valid-tokens','1000000'],
    stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print("DATA_PREP_SPAWNED",p.pid,flush=True)
