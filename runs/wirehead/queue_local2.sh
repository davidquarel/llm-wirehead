#!/bin/bash
# Seed-1 replicates on the local 5090: Qwen3.5-4B then Qwen3.5-9B, 40 steps each.
# Start: setsid nohup bash runs/wirehead/queue_local2.sh > runs/wirehead/queue_local2.out 2>&1 &
cd ~/Dropbox/ARENA/weight_space_prog
for q in "Qwen/Qwen3.5-4B|runs/wirehead/qwen3.5-4b-v3-seed1" "Qwen/Qwen3.5-9B|runs/wirehead/qwen3.5-9b-v3-seed1"; do
  IFS='|' read -r model run <<< "$q"
  echo "$(date -u +%H:%M) start $model -> $run"
  MODEL=$model RUN=$run LAYERS="4 8 12 16 20 24 28" STEPS=40 bash runs/wirehead/launch_local.sh --seed 1
  while kill -0 $(cat $run/wirehead.pid) 2>/dev/null; do sleep 60; done
  echo "$(date -u +%H:%M) agent exited: $(tail -1 $run/wirehead.out | cut -c1-150)"
  kill $(cat $run/server.pid) 2>/dev/null; sleep 15
done
