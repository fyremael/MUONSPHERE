#!/bin/bash
set -u
seed="$1"
sess="gcl-fsm-repl-s${seed}-1021"
dest="/home/jim/gcl-research/attention-sink-s4/repo/experiments/tnlm_replication/receipts/seed${seed}"
mkdir -p "$dest"
remotedir=/content/gcl_fsm_three_seed
if [ "$seed" = "101" ]; then remotedir=/content/gcl_fsm_seed101; sess=gcl-fsm-repl-s202-1021; fi
for i in $(seq 1 320); do
 timeout 18s colab download -s "$sess" $remotedir/metrics.jsonl "$dest/metrics.jsonl" > /dev/null 2>&1 || true
 for name in summary.json retrieval.jsonl initial.jsonl; do
   timeout 18s colab download -s "$sess" "$remotedir/$name" "$dest/$name" > /dev/null 2>&1 || true
 done
 if timeout 18s colab download -s "$sess" $remotedir/run_status.json "$dest/run_status.json" > /dev/null 2>&1; then
   for name in config.json run.log sources.sha256; do
      timeout 20s colab download -s "$sess" "$remotedir/$name" "$dest/$name" >> "$dest/watcher.log" 2>&1 || true
   done
   echo "DONE $(date -u +%T)" >> "$dest/watcher.log"
   exit 0
 fi
 echo "WATCH $i $(date -u +%T)" >> "$dest/watcher.log"
 sleep 15
done
echo "MISSING_COMPLETION" >> "$dest/watcher.log"
