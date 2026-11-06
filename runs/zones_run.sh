#!/bin/bash
# E1 zones on arena9-trouble: Llama-3.1-8B then Qwen3.5-4B (96 sessions each). setsid nohup bash zones_run.sh > zones_run.log 2>&1 &
cd /root/wsp-dq
export HF_HOME=/root/wsp-dq/data/hf PATH=/root/wsp-dq/venv/bin:$PATH OMP_NUM_THREADS=4
for m in "unsloth/Llama-3.1-8B-Instruct|runs/zones/llama8b" "Qwen/Qwen3.5-4B|runs/zones/qwen4b"; do
  IFS='|' read -r model out <<< "$m"
  echo "$(date -u +%H:%M) gen $model"; venv/bin/python zones.py gen --model $model --out $out --n 96 2>&1 | grep -E "turn|Error|Trace"
  echo "$(date -u +%H:%M) steer $model"; venv/bin/python zones.py steer --model $model --out $out 2>&1 | grep -vE "Loading weights|Warning"
done
echo "$(date -u +%H:%M) all done"
