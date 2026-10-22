#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/fromscratch_a100
for i in $(seq 1 250); do
  timeout 15s colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_comparison/metrics.jsonl compare-live.jsonl >/dev/null 2>&1 || true
  timeout 15s colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_comparison/run.log compare-live.log >/dev/null 2>&1 || true
  if timeout 15s colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_comparison/run_status.json compare-status.json >/dev/null 2>&1; then
    timeout 20s colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_comparison/summary.json compare-summary.json >/dev/null 2>&1 || true
    timeout 20s colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_comparison/config.json compare-config.json >/dev/null 2>&1 || true
    timeout 20s colab download -s gcl-s4-scratch-retry-1021 /content/gcl_s4_scratch_comparison/initial.jsonl compare-initial.jsonl >/dev/null 2>&1 || true
    echo "COMPLETE $(date -u +%T)" >> compare-watcher.log
    exit 0
  fi
  echo "POLL $i $(date -u +%T)" >> compare-watcher.log
  sleep 10
done
echo "WATCH_LIMIT" >> compare-watcher.log
