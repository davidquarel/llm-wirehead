#!/bin/bash
# wirehead runs on nicky1234 (3x RTX A4000 16 GB, 31 GB RAM: at most 2 servers at once). Copy to ~/weight_space_prog/.
#   GPUS=0 PORT=6211 MODEL=Qwen/Qwen3.5-4B RUN=runs/wirehead/qwen3.5-4b-v3 LAYERS="4 8 12 16 20 24 28" bash launch_nicky.sh [--resume]
#   GPUS=1,2 TP=2 PORT=6212 MODEL=unsloth/Llama-3.1-8B-Instruct RUN=runs/wirehead/llama3.1-8b-v3 LAYERS="..." bash launch_nicky.sh
cd ~/weight_space_prog
export HF_HOME=~/weight_space_prog/hf PATH=~/weight_space_prog/venv/bin:$PATH CUDA_VISIBLE_DEVICES=$GPUS
d=$RUN; mkdir -p $d/server
if ! grep -q "^ready on" $d/server.out 2>/dev/null || ! kill -0 $(cat $d/server.pid 2>/dev/null) 2>/dev/null; then
  OMP_NUM_THREADS=2 setsid nohup venv/bin/python bliss_server.py --model $MODEL --run $d/server --port $PORT --tp ${TP:-1} \
      --gpu-mem ${GPU_MEM:-0.88} --max-model-len 8192 --max-num-seqs ${MAX_SEQS:-64} > $d/server.out 2>&1 < /dev/null &
  echo $! > $d/server.pid
  for i in $(seq 240); do grep -q "^ready on" $d/server.out && break; kill -0 $(cat $d/server.pid) || exit 1; sleep 5; done
fi
OMP_NUM_THREADS=1 setsid nohup venv/bin/python wirehead3.py --port $PORT --run $d --layers $LAYERS --steps ${STEPS:-40} \
    --chunk ${CHUNK:-64} "$@" >> $d/wirehead.out 2>&1 < /dev/null &
echo $! > $d/wirehead.pid
