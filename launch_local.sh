#!/bin/bash
# usage: launch_local.sh RUN_DIR PORT  -- resume a 27B self-edit run on the local GPU (server, then agent)
cd /root/Dropbox/ARENA/weight_space_prog
export HF_HOME=/root/wsp/hf PATH=/root/wsp/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
run=$1; port=$2
setsid nohup python selfedit_server.py --run "$run" --budget 120 --port "$port" --gpu-mem 0.80 --max-model-len 65536 --max-num-seqs 64 > "$run.log" 2>&1 < /dev/null &
echo $! > "$run/server.pid"
until grep -qE "^ready|Traceback" "$run.log"; do sleep 5; done
grep -qE "^ready" "$run.log" || { tail -5 "$run.log"; exit 1; }
setsid nohup python agent.py --run "$run" --port "$port" --self-edit --no-thinking --explicit-submit --max-tokens 8192 --max-ctx-chars 140000 --budget 120 --turns 300 --prompt prompt_self_v2.md --resume >> "$run/agent.out" 2>&1 < /dev/null &
echo $! > "$run/agent.pid"
echo "server $(cat $run/server.pid) agent $(cat $run/agent.pid)"
