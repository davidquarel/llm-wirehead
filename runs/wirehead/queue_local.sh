#!/bin/bash
# Attractor survey on the local 5090: one wirehead3 run per model, STEPS steps each, one after another.
# Start: setsid nohup bash runs/wirehead/queue_local.sh > runs/wirehead/queue_local.out 2>&1 &
# Each run can be resumed by hand with launch_local.sh --resume. Edit QUEUE to change the order.
cd ~/Dropbox/ARENA/weight_space_prog
STEPS=${STEPS:-40}
step_of() { python3 -c "import json,sys;print(max([json.loads(l).get('step',0) for l in open('$1/wirehead.jsonl') if '\"step\"' in l] or [0]))" 2>/dev/null || echo 0; }
stop_run() { kill $(cat $1/wirehead.pid) 2>/dev/null; kill $(cat $1/server.pid) 2>/dev/null; sleep 15; }
# 1. let the running 4B reach STEPS, then stop it
d=runs/wirehead/qwen3.5-4b-v3
while kill -0 $(cat $d/wirehead.pid) 2>/dev/null && [ $(step_of $d) -lt $STEPS ]; do sleep 60; done
stop_run $d; echo "$(date -u +%H:%M) done $d"
QUEUE=(
  "Qwen/Qwen3.5-9B|runs/wirehead/qwen3.5-9b-v3|4 8 12 16 20 24 28"
)
for q in "${QUEUE[@]}"; do
  IFS='|' read -r model run layers <<< "$q"
  echo "$(date -u +%H:%M) start $model -> $run"
  MODEL=$model RUN=$run LAYERS="$layers" STEPS=$STEPS bash runs/wirehead/launch_local.sh
  while kill -0 $(cat $run/wirehead.pid) 2>/dev/null; do sleep 60; done
  echo "$(date -u +%H:%M) agent exited at step $(step_of $run): $(tail -1 $run/wirehead.out | cut -c1-200)"
  stop_run $run
done
