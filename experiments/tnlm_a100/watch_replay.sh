#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_a100
sess=gcl-fsm-replay-a100-1021
for i in $(seq 1 100); do
 timeout 18s colab download -s "$sess" /content/gcl_fsm_replay/metrics.jsonl replay-metrics.jsonl >/dev/null 2>&1 || true
 timeout 18s colab download -s "$sess" /content/gcl_fsm_replay/run.log replay-run.log >/dev/null 2>&1 || true
 if timeout 18s colab download -s "$sess" /content/gcl_fsm_replay/controller_status.json replay-controller-status.json >/dev/null 2>&1; then
   for name in config.json initial.jsonl metrics.jsonl summary.json run_status.json; do
     timeout 20s colab download -s "$sess" "/content/gcl_fsm_replay/$name" "replay-$name" >/dev/null 2>&1 || true
   done
   timeout 18s colab download -s "$sess" /content/gcl_tnlm_fineweb/corpus_manifest.json replay-corpus-manifest.json >/dev/null 2>&1 || true
   echo "CONTROLLER_COMPLETED $i $(date -u +%FT%TZ)" >> replay-watch.log
   exit 0
 fi
 echo "POLL $i $(date -u +%FT%TZ)" >> replay-watch.log
 sleep 15
done
echo WATCH_LIMIT >> replay-watch.log
