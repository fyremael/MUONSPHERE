#!/bin/bash
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_a100
sess=gcl-tnlm-sink-a100-1021
for iteration in $(seq 1 160); do
    timeout 15s colab download -s "$sess" /content/gcl_tnlm_induction/metrics.jsonl induction-metrics.jsonl >/dev/null 2>&1 || true
    timeout 15s colab download -s "$sess" /content/gcl_tnlm_induction/run.log induction-run.log >/dev/null 2>&1 || true
    if timeout 15s colab download -s "$sess" /content/gcl_tnlm_induction/run_status.json induction-status.json >/dev/null 2>&1; then
        for name in summary.json config.json initial.jsonl; do
            timeout 20s colab download -s "$sess" "/content/gcl_tnlm_induction/$name" "induction-$name" >/dev/null 2>&1 || true
        done
        echo "COMPLETED poll=$iteration $(date -u +%FT%TZ)" >> induction-watch.log
        exit 0
    fi
    echo "POLL $iteration $(date -u +%FT%TZ)" >> induction-watch.log
    sleep 15
done
echo WATCH_LIMIT >> induction-watch.log
