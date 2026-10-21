import pathlib,subprocess,sys
d=pathlib.Path('/content/gcl_s4_scratch_corpus');d.mkdir(parents=True,exist_ok=True)
f=open(d/'prepare.log','w',buffering=1)
cmd=[sys.executable,'/content/gcl_s4_scratch_prepare.py',
'--train-tokens','60000000','--valid-tokens','4000000','--out',str(d)]
p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('PREP_STARTED',p.pid)
