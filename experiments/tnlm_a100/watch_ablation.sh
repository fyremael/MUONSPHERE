#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_a100
session=gcl-tnlm-sink-a100-1021
for iteration in $(seq 1 200); do
  timeout 15s colab download -s "$session" \
    /content/gcl_tnlm_s4_ablation/metrics.jsonl ablation-metrics.jsonl >/dev/null 2>&1 || true
  timeout 15s colab download -s "$session" \
    /content/gcl_tnlm_s4_ablation/run.log ablation-run.log >/dev/null 2>&1 || true
  if timeout 15s colab download -s "$session" \
     /content/gcl_tnlm_s4_ablation/run_status.json ablation-status.json >/dev/null 2>&1; then
    for name in config.json summary.json initial.jsonl; do
      timeout 20s colab download -s "$session" \
        "/content/gcl_tnlm_s4_ablation/$name" "ablation-$name" >/dev/null 2>&1 || true
    done
    echo "COMPLETED poll=$iteration utc=$(date -u +%FT%TZ)" >> ablation-watch.log
    exit 0
  fi
  echo "POLL $iteration utc=$(date -u +%FT%TZ)" >> ablation-watch.log
  sleep 15
done
echo WATCH_LIMIT >> ablation-watch.log
