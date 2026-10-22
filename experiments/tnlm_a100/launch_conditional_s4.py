"""Launch S4 comparisons only if TNLM or force-sink arm passed induction.

Never call absence of induced sink prevention. Reuses identical FineWeb
corpus, seed, schedule; the appropriate matched S4 controls are selected
from exact 4000-step induction metrics.
"""
import json,subprocess,sys,pathlib
source=pathlib.Path('/content/gcl_tnlm_induction')
status=json.loads((source/'run_status.json').read_text())
assert status['complete'] and status['count']==status['expected']==2,status
records=[json.loads(x) for x in (source/'metrics.jsonl').read_text().splitlines()]
def gate(arm):
    rows=sorted([r for r in records if r['arm']==arm and r['step']>1],
                key=lambda x:x['step'])
    assert rows[-1]['step']==4000
    for layer in range(12):
        for start in range(len(rows)-2):
            triple=rows[start:start+3]
            if all(r['sink_fraction_by_layer'][layer]>=0.25 for r in triple):
                return {'induced':True,'layer':layer,
                        'start_step':triple[0]['step'],
                        'end_step':triple[-1]['step']}
    return {'induced':False,'max_layer_fraction':
       max(max(r['sink_fraction_by_layer']) for r in rows)}
tnlm=gate('tnlm')
fsm=gate('fsm_tnlm')
if tnlm['induced']:
    selected=['tnlm_s4','tnlm_s4_col']
    prefix='tnlm'
elif fsm['induced']:
    selected=['fsm_tnlm_s4','fsm_tnlm_s4_col']
    prefix='fsm_tnlm'
else:
    selected=[]
    prefix=None
packet={'record_type':'GCL_TNLM_S4_DISPATCH_DECISION/1',
        'tnlm':tnlm,'fsm_tnlm':fsm,'selected_arms':selected,
        'requires_induction_gate':True}
(source/'dispatch_decision.json').write_text(json.dumps(packet,indent=2))
print("INDUCTION_DISPATCH_DECISION",json.dumps(packet,sort_keys=True),flush=True)
if not selected:
    print("BLOCKED_NO_PRONOUNCED_SINK",flush=True)
    sys.exit(0)
out=pathlib.Path('/content/gcl_tnlm_s4_ablation')
out.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'/content/gcl_tnlm_train.py',
 '--data','/content/gcl_tnlm_fineweb','--out',str(out),
 '--stage','tnlm_matched_s4_after_gate',
 '--context','512','--batch','2','--accum','4',
 '--steps','4000','--warmup','100','--peak-lr','0.0003',
 '--eval-every','250','--eval-batches','4',
 '--seeds','101','--arms'] + selected + [
 '--force-sink-k','16','--lambda-s4','0.15','--lambda-column','0.15',
 '--long-eval','--wandb','offline']
with open(out/'run.log','w',buffering=1) as file:
    process=subprocess.Popen(cmd,stdout=file,stderr=subprocess.STDOUT,
                             start_new_session=True)
print('S4_ABLATION_STARTED',json.dumps({'pid':process.pid,'arms':selected,
 'prefix':prefix}),flush=True)
