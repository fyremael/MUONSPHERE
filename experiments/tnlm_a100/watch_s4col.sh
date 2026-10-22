#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_a100
sess=gcl-fsm-s4col-a100-1021
for i in $(seq 1 100); do
 timeout 18s colab download -s "$sess" /content/gcl_fsm_s4_col/metrics.jsonl s4col-metrics.jsonl >/dev/null 2>&1 || true
 timeout 18s colab download -s "$sess" /content/gcl_fsm_s4_col/run.log s4col-run.log >/dev/null 2>&1 || true
 if timeout 18s colab download -s "$sess" /content/gcl_fsm_s4_col/controller_status.json s4col-controller-status.json >/dev/null 2>&1; then
   for name in config.json initial.jsonl metrics.jsonl summary.json run_status.json; do
     timeout 20s colab download -s "$sess" "/content/gcl_fsm_s4_col/$name" "s4col-$name" >/dev/null 2>&1 || true
   done
   timeout 18s colab download -s "$sess" /content/gcl_tnlm_fineweb/corpus_manifest.json s4col-corpus-manifest.json >/dev/null 2>&1 || true
   echo "CONTROLLER_COMPLETED $i $(date -u +%FT%TZ)" >> s4col-watch.log
   exit 0
 fi
 echo "POLL $i $(date -u +%FT%TZ)" >> s4col-watch.log
 sleep 15
done
echo WATCH_LIMIT >> s4col-watch.log
