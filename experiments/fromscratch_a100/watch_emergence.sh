#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/fromscratch_a100
for i in $(seq 1 100); do
  colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_emergence/metrics.jsonl emergence-live.jsonl > /dev/null 2>&1 || true
  colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_emergence/run.log emergence-live.log > /dev/null 2>&1 || true
  if colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_emergence/run_status.json emergence-status.json > /dev/null 2>&1; then
    colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_emergence/summary.json emergence-summary.json > /dev/null 2>&1 || true
    colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_emergence/config.json emergence-config.json > /dev/null 2>&1 || true
    echo "COMPLETE $(date -u +%T)" >> watcher.log
    exit 0
  fi
  echo "POLL $i $(date -u +%T)" >> watcher.log
  sleep 10
done
echo "WATCHER_LIMIT" >> watcher.log
