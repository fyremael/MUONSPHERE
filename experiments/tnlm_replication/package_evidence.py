#!/usr/bin/env python3
"""Reproducibility package for completed three-seed S4 retrieval study."""
import hashlib,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=REPO/'docs/data'
ANALYSIS=OUT/'tnlm_fsm_three_seed_retrieval_analysis.json'
REPORT=REPO/'docs/attention_sink_tnlm_fsm_three_seed_retrieval_report.md'
SEEDS=(101,202,303)
SOURCE=(
 'experiments/tnlm_replication/README.md',
 'experiments/tnlm_replication/prepare_fineweb.py',
 'experiments/tnlm_replication/train_replicate.py',
 'experiments/tnlm_replication/retrieval_eval.py',
 'experiments/tnlm_replication/analyze.py',
 'experiments/tnlm_replication/package_evidence.py',
 'experiments/tnlm_replication/archive_private.py',
 'experiments/tnlm_replication/watch_seed.sh',
 'docs/attention_sink_tnlm_fsm_three_seed_retrieval_report.md'
)

def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    result=json.loads(ANALYSIS.read_text())
    assert result['record_type']=='GCL_FSM_S4_THREE_SEED_RETRIEVAL/1'
    assert result['status']=='COMPLETE'
    files={}
    for name in SOURCE:
        p=REPO/name
        assert p.exists(),p
        files['source/'+name]=p.read_bytes()
    files['analysis.json']=ANALYSIS.read_bytes()
    common_sha=set()
    for seed in SEEDS:
        root=HERE/'receipts'/f'seed{seed}'
        for name in ('config.json','metrics.jsonl','initial.jsonl','retrieval.jsonl',
                     'summary.json','run_status.json','sources.sha256'):
            p=root/name
            assert p.exists(),p
            files[f'observations/seed{seed}/{name}']=p.read_bytes()
        hashes=dict(line.split(' ',1) for line in
                    (root/'sources.sha256').read_text().splitlines())
        assert hashes[sha(files['source/experiments/tnlm_replication/train_replicate.py'])]=='/content/gcl_rep_train.py'
        assert hashes[sha(files['source/experiments/tnlm_replication/retrieval_eval.py'])]=='/content/retrieval_eval.py'
        config=json.loads((root/'config.json').read_text())
        common_sha.add(config['corpus']['manifest_sha256'])
    assert len(common_sha)==1
    manifest=dict(
      record_type='GCL_FSM_S4_REPLICATION_RETRIEVAL_EVIDENCE/1',
      status='complete',
      seeds=list(SEEDS),arm_count=9,step_count_each=4000,
      model='124M GPT2 random initialization',
      natural_text='FineWeb-Edu sample-10BT',
      attention_logit_bias=False,
      structurally_induced=True,
      corpus_manifest_sha256=next(iter(common_sha)),
      receipt_sha256={n:sha(data) for n,data in sorted(files.items())},
      private_wandb_binaries_not_public=True,
      note='Context-copy and passkey tests are zero-shot, and must not be '
           'confused with supervised retrieval proficiency')
    files['manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    out=OUT/'tnlm_fsm_three_seed_retrieval_bundle.zip'
    with ZipFile(out,'w',ZIP_DEFLATED) as z:
        for n,data in sorted(files.items()):z.writestr(n,data)
    with ZipFile(out) as z:
        for n,h in manifest['receipt_sha256'].items():
            assert sha(z.read(n))==h,n
    (OUT/'tnlm_fsm_three_seed_retrieval_manifest.json').write_bytes(files['manifest.json'])
    print('PUBLISHED_BUNDLE',out,'bytes',out.stat().st_size,
          'sha256',sha(out.read_bytes()),'files',len(files))
if __name__=='__main__':main()
