"""Reproducibly reconstruct corpus then run a single 4000-step FSM arm.

The controller survives CLI disconnect. It does not count an incomplete
earlier run toward any evidence of completion.
"""
from pathlib import Path
import subprocess,sys,json,traceback,time
root=Path('/content')
data=root/'gcl_tnlm_fineweb'
data.mkdir(parents=True,exist_ok=True)
log=root/'gcl_fsm_replay'
log.mkdir(parents=True,exist_ok=True)
def execute(cmd,name):
    with (log/name).open('w',buffering=1) as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=False)
    if r.returncode:
        raise RuntimeError(f'{name} exited {r.returncode}')
try:
    execute([sys.executable,'/content/gcl_tnlm_prepare.py',
        '--out',str(data),'--train-tokens','60000000','--valid-tokens','4000000'],'prepare.log')
    manifest=json.loads((data/'corpus_manifest.json').read_text())
    assert manifest['manifest_sha256']=='5233f44b9091a9d4f0af9f5abebfc6bcb64d4da098f7cfddf52bfe91630acbba'
    assert manifest['train_sha256']=='953b7aa03a10181db8025c6170acacfb3e242e9978cdf0892b327ab937469552'
    assert manifest['valid_sha256']=='3f158048b74095ac5446cabf8f66596b1aae3b5907c3289e107d20228d25a43a'
    print('REPLAY_CORPUS_VERIFIED',flush=True)
    execute([sys.executable,'/content/gcl_tnlm_train.py',
        '--data',str(data),'--out',str(log),
        '--stage','FSM_single_arm_full_replay',
        '--context','512','--batch','2','--accum','4',
        '--steps','4000','--warmup','100','--peak-lr','0.0003',
        '--eval-every','250','--eval-batches','4',
        '--seeds','101','--arms','fsm_tnlm','--force-sink-k','16',
        '--long-eval','--wandb','offline'],'run.log')
    status=json.loads((log/'run_status.json').read_text())
    assert status['complete'] and status['count']==1
    (log/'controller_status.json').write_text(json.dumps(
        {'status':'COMPLETE','arm':'fsm_tnlm','steps':4000},indent=2))
    print('REPLAY_COMPLETE',flush=True)
except Exception as exc:
    (log/'controller_status.json').write_text(json.dumps(
        {'status':'ERROR','message':repr(exc),
         'traceback':traceback.format_exc()},indent=2))
    print('REPLAY_ERROR',repr(exc),flush=True)
    raise
