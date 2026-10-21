"""Admit S4-only direct execution after verified cached-corpus A100 run.

Preserve the prior failed-controller receipt and only emit a completion
receipt if all complete-run, exact-initialization and provenance checks pass.
"""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent

def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def finalize():
    status=json.loads((HERE/'s4only-run_status.json').read_text())
    assert status['complete'] and status['count']==status['expected']==1,status
    cfg=json.loads((HERE/'s4only-config.json').read_text())
    assert cfg['arms']==['fsm_tnlm_s4']
    assert cfg['seeds']==[101] and cfg['steps']==4000
    replay=json.loads((HERE/'replay-config.json').read_text())
    col=json.loads((HERE/'s4col-config.json').read_text())
    for key in ('batch','accum','context','steps','eval_every','eval_batches',
                'warmup','peak_lr','lambda_s4','lambda_column'):
        assert cfg[key]==replay[key]==col[key],key
    for key in ('train_sha256','valid_sha256','manifest_sha256'):
        assert cfg['corpus'][key]==replay['corpus'][key]==col['corpus'][key],key
    initial=jl(HERE/'s4only-initial.jsonl')
    assert len(initial)==1
    replay_init=jl(HERE/'replay-initial.jsonl')
    col_init=jl(HERE/'s4col-initial.jsonl')
    assert (initial[0]['model_sha256_sample']==replay_init[0]['model_sha256_sample']
            ==col_init[0]['model_sha256_sample'])
    rows=jl(HERE/'s4only-metrics.jsonl')
    assert rows[0]['step']==1 and rows[-1]['step']==4000
    assert all(r['arm']=='fsm_tnlm_s4' and r['seed']==101 for r in rows)
    summary=json.loads((HERE/'s4only-summary.json').read_text())
    assert len(summary)==1 and summary[0]['completed']
    assert summary[0]['step']==4000
    assert summary[0]['last']['step']==4000
    assert len(rows)==17, len(rows)
    old=HERE/'s4only-controller-status.json'
    failed=HERE/'s4only-failed-controller-status.json'
    if old.exists() and json.loads(old.read_text()).get('status')=='ERROR':
        if not failed.exists():failed.write_bytes(old.read_bytes())
    receipt={'status':'COMPLETE','arm':'fsm_tnlm_s4','steps':4000,
             'execution':'DIRECT_CACHED_CORPUS_A100',
             'earlier_failed_prepare_receipt':failed.name if failed.exists() else None,
             'metrics_sha256':hashlib.sha256((HERE/'s4only-metrics.jsonl').read_bytes()).hexdigest(),
             'initial_sample_sha256':initial[0]['model_sha256_sample'],
             'corpus_manifest_sha256':cfg['corpus']['manifest_sha256']}
    old.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('S4_ONLY_VERIFIED',json.dumps(receipt))
if __name__=='__main__':finalize()
