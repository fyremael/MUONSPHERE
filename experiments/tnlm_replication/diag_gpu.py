import subprocess,time
r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
print('GPU_STATE',r.stdout.strip(),flush=True)
for row in subprocess.run(['ps','-eo','pid,etimes,pcpu,pmem,args'],capture_output=True,text=True).stdout.splitlines():
 if 'gcl_rep_train.py' in row:print('TRAINING_PROCESS',row[:280])
