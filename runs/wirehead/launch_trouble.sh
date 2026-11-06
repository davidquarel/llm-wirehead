#!/bin/bash
# Wirehead loops on the 27B, arena9-trouble (RTX PRO 6000 96 GB). Copy to /root/wsp-dq/ there and run:
#   RUN=runs/wirehead/qwen3.8-27b-v3 SCRIPT=wirehead3.py bash launch_trouble.sh [--resume]
# The vLLM server lives in $SRV (default: the v2 run dir, where it was first started) and is shared by every run.
cd /root/wsp-dq
export HF_HOME=/root/wsp-dq/data/hf PATH=/root/wsp-dq/venv/bin:$PATH
s=${SRV:-runs/wirehead/qwen3.8-27b-v2}; d=${RUN:-runs/wirehead/qwen3.8-27b-v3}; mkdir -p $s/server $d
if ! grep -q "^ready on" $s/server.out 2>/dev/null || ! kill -0 $(cat $s/server.pid 2>/dev/null) 2>/dev/null; then
  OMP_NUM_THREADS=4 setsid nohup venv/bin/python bliss_server.py --model RedHatAI/Qwen3.8-27B-INT4 --run $s/server --port 6203 \
      --gpu-mem 0.88 --max-model-len 16384 --max-num-seqs 256 > $s/server.out 2>&1 < /dev/null &
  echo $! > $s/server.pid
  for i in $(seq 240); do grep -q "^ready on" $s/server.out && break; kill -0 $(cat $s/server.pid) || exit 1; sleep 5; done
fi
OMP_NUM_THREADS=1 setsid nohup venv/bin/python ${SCRIPT:-wirehead3.py} --port 6203 --run $d --layers 16 24 32 40 48 --steps 150 --chunk 384 "$@" \
    >> $d/wirehead.out 2>&1 < /dev/null &
echo $! > $d/wirehead.pid
