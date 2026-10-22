#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_a100
sess=gcl-fsm-s4only-a100-1021
for i in $(seq 1 100); do
 timeout 18s colab download -s "$sess" /content/gcl_fsm_s4_only/metrics.jsonl s4only-metrics.jsonl >/dev/null 2>&1 || true
 timeout 18s colab download -s "$sess" /content/gcl_fsm_s4_only/run.log s4only-run.log >/dev/null 2>&1 || true
 if timeout 18s colab download -s "$sess" /content/gcl_fsm_s4_only/controller_status.json s4only-controller-status.json >/dev/null 2>&1; then
   for name in config.json initial.jsonl metrics.jsonl summary.json run_status.json; do
     timeout 20s colab download -s "$sess" "/content/gcl_fsm_s4_only/$name" "s4only-$name" >/dev/null 2>&1 || true
   done
   timeout 18s colab download -s "$sess" /content/gcl_tnlm_fineweb/corpus_manifest.json s4only-corpus-manifest.json >/dev/null 2>&1 || true
   echo "CONTROLLER_COMPLETED $i $(date -u +%FT%TZ)" >> s4only-watch.log
   exit 0
 fi
 echo "POLL $i $(date -u +%FT%TZ)" >> s4only-watch.log
 sleep 15
done
echo WATCH_LIMIT >> s4only-watch.log
