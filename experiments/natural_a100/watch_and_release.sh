#!/bin/bash
# Robust retrieval and cleanup of the *already dispatched* Colab compute.
set -u
cd /home/jim/gcl-research/attention-sink-s4/repo/experiments/natural_a100
sess=gcl-natural-sink-a100-1021
for cycle in $(seq 1 90); do
  if colab download -s "$sess" /content/gcl_natural_sink_full/run_status.json full-status.json >/dev/null 2>&1; then
    echo "FOUND_RUN_STATUS cycle=$cycle" >> watcher.log
    colab download -s "$sess" /content/gcl_natural_sink_full/summary.json full-summary.json >> watcher.log 2>&1 || true
    colab download -s "$sess" /content/gcl_natural_sink_full/metrics.jsonl full-metrics.jsonl >> watcher.log 2>&1 || true
    colab download -s "$sess" /content/gcl_natural_sink_full/initial_metrics.jsonl full-initial.jsonl >> watcher.log 2>&1 || true
    colab download -s "$sess" /content/gcl_natural_sink_full/config.json full-config.json >> watcher.log 2>&1 || true
    colab download -s "$sess" /content/gcl_natural_sink/corpus_manifest.json corpus-manifest.json >> watcher.log 2>&1 || true
    colab download -s "$sess" /content/gcl_natural_sink_full/run.log full-colab.log >> watcher.log 2>&1 || true
    break
  fi
  if [ "$cycle" -eq 90 ]; then
    echo TIMEOUT_NO_COMPLETE_STATUS >> watcher.log
    colab download -s "$sess" /content/gcl_natural_sink_full/run.log full-colab.log >> watcher.log 2>&1 || true
    colab download -s "$sess" /content/gcl_natural_sink_full/summary.json full-summary.json >> watcher.log 2>&1 || true
    colab download -s "$sess" /content/gcl_natural_sink_full/metrics.jsonl full-metrics.jsonl >> watcher.log 2>&1 || true
  fi
  sleep 22
done
colab stop -s "$sess" >> watcher.log 2>&1 || true
echo RELEASED >> watcher.log
