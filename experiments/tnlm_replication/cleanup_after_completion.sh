#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_replication
finish_one() {
 local session="$1" outfile="$2"
 for attempt in $(seq 1 7); do
  if timeout 100s colab exec -s "$session" -f archive_private.py --timeout 80 >> cleanup.log 2>&1 &&
    timeout 70s colab download -s "$session" /content/gcl_three_seed_private_receipts.zip "$outfile" >> cleanup.log 2>&1; then
    if python3 - "$outfile" <<'PY'
import json,sys,zipfile
p=sys.argv[1]
with zipfile.ZipFile(p) as z:
    names=z.namelist()
    assert any(n.endswith('.wandb') for n in names)
    assert len([n for n in names if n.endswith('run_status.json')]) in (1,2)
    for n in names:
        if n.endswith('run_status.json'):
            s=json.loads(z.read(n));assert s['complete'] and s['count']==s['expected']==3
print('ARCHIVE_AUDITED',p,len(names))
PY
    then
      echo "ARCHIVE_VERIFIED $session" >> cleanup.log
      timeout 30s colab stop -s "$session" >> cleanup.log 2>&1 || true
      return 0
    fi
  fi
  echo "ARCHIVE_RETRY $session $attempt" >> cleanup.log
  sleep 25
 done
 echo "RECOVERY_REQUIRED $session: private W&B archive could not be validated; GPU still allocated" >> cleanup.log
 return 1
}
for cycle in $(seq 1 420); do
 if [ -f receipts/seed303/run_status.json ]; then
   echo "GOT_FINAL_SEED303 $(date -u +%T)" >> cleanup.log
   finish_one gcl-fsm-repl-s303-1021 receipts/seed303/private_archive.zip || true
   break
 fi
 sleep 20
done
for cycle in $(seq 1 420); do
 if [ -f receipts/seed202/run_status.json ] && [ -f receipts/seed101/run_status.json ]; then
   echo "GOT_FINAL_BOTH_SHARED $(date -u +%T)" >> cleanup.log
   finish_one gcl-fsm-repl-s202-1021 receipts/seed202/private_shared_archive.zip || true
   break
 fi
 sleep 20
done
echo "AUTOMATIC_CUSTODY_SWEEP_COMPLETE $(date -u +%T)" >> cleanup.log
