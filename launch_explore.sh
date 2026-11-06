#!/bin/bash
# usage: launch_explore.sh MODEL RUN_DIR PORT [extra explore_server args]  (GPUs from CUDA_VISIBLE_DEVICES)
cd ~/weight_space_prog
export HF_HOME=~/weight_space_prog/hf VLLM_USE_FLASHINFER_SAMPLER=0
export PATH=~/weight_space_prog/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
model=$1; run=$2; port=$3; shift 3
mkdir -p "$run"
setsid nohup venv/bin/python explore_server.py --model "$model" --run "$run" --port "$port" "$@" > "$run.log" 2>&1 < /dev/null &
echo $! > "$run.pid"
for i in $(seq 400); do grep -qE "^ready|Traceback" "$run.log" && break; sleep 3; done
grep -E "^ready|Error" "$run.log" | tail -3 | cut -c1-250
