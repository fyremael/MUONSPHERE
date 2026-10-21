#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_a100
sess=gcl-fsm-s4only-a100-1021
for i in $(seq 1 160); do
 timeout 15 colab download -s "$sess" /content/gcl_fsm_s4only_direct/metrics.jsonl s4only-metrics.jsonl >/dev/null 2>&1 || true
 if timeout 15 colab download -s "$sess" /content/gcl_fsm_s4only_direct/run_status.json s4only-run_status.json >/dev/null 2>&1; then
  for name in config.json initial.jsonl metrics.jsonl summary.json run.log; do
   timeout 35 colab download -s "$sess" "/content/gcl_fsm_s4only_direct/$name" "s4only-$name" >> s4only-direct-watch.log 2>&1 || true
  done
  echo "COMPLETE $i" >> s4only-direct-watch.log
  exit 0
 fi
 echo "POLL $i $(date -u +%T)" >> s4only-direct-watch.log
 sleep 10
done
echo "TIMEOUT" >> s4only-direct-watch.log
