#!/bin/bash
# Generic local launcher (5090, port 6203). Usage: MODEL=Qwen/Qwen3.5-4B RUN=runs/wirehead/qwen3.5-4b-v3 SCRIPT=wirehead3.py \
#   LAYERS="4 8 12 16 20 24 28" bash runs/wirehead/launch_local.sh [--resume] [extra args]
cd ~/Dropbox/ARENA/weight_space_prog
export HF_HOME=data/hf
d=$RUN; mkdir -p $d/server
if ! grep -q "^ready on" $d/server.out 2>/dev/null || ! kill -0 $(cat $d/server.pid 2>/dev/null) 2>/dev/null; then
  OMP_NUM_THREADS=4 setsid nohup data/venv/bin/python bliss_server.py --model $MODEL --run $d/server --port 6203 --gpu-mem ${GPU_MEM:-0.85} \
      --max-model-len 16384 --max-num-seqs ${MAX_SEQS:-128} > $d/server.out 2>&1 < /dev/null &
  echo $! > $d/server.pid
  for i in $(seq 180); do grep -q "^ready on" $d/server.out && break; sleep 5; done
fi
OMP_NUM_THREADS=1 setsid nohup data/venv/bin/python ${SCRIPT:-wirehead3.py} --port 6203 --run $d --layers $LAYERS --steps ${STEPS:-150} "$@" \
    >> $d/wirehead.out 2>&1 < /dev/null &
echo $! > $d/wirehead.pid
