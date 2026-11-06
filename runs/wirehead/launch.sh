#!/bin/bash
# Self-wireheading loop on Qwen3.5-4B (5090, port 6203). Restart: bash runs/wirehead/launch.sh [--resume]
cd ~/Dropbox/ARENA/weight_space_prog
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}   # shared cache (was data/hf until 2026-11-06)
d=runs/wirehead/qwen3.5-4b; mkdir -p $d/server
if ! grep -q "^ready on" $d/server.out 2>/dev/null || ! kill -0 $(cat $d/server.pid 2>/dev/null) 2>/dev/null; then
  OMP_NUM_THREADS=4 setsid nohup data/venv/bin/python bliss_server.py --model Qwen/Qwen3.5-4B --run $d/server --port 6203 --gpu-mem 0.85 \
      --max-model-len 16384 --max-num-seqs 128 > $d/server.out 2>&1 < /dev/null &
  echo $! > $d/server.pid
  for i in $(seq 120); do grep -q "^ready on" $d/server.out && break; sleep 5; done
fi
OMP_NUM_THREADS=1 setsid nohup data/venv/bin/python wirehead.py --port 6203 --run $d --layers 4 8 12 16 20 24 28 --steps 60 "$@" \
    >> $d/wirehead.out 2>&1 < /dev/null &
echo $! > $d/wirehead.pid
