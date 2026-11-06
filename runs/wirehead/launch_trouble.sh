#!/bin/bash
# wirehead2 on the 27B, arena9-trouble (RTX PRO 6000 96 GB). Copy to /root/wsp-dq/ there and run: bash launch_trouble.sh
cd /root/wsp-dq
export HF_HOME=/root/wsp-dq/data/hf
d=runs/wirehead/qwen3.8-27b-v2; mkdir -p $d/server
if ! grep -q "^ready on" $d/server.out 2>/dev/null || ! kill -0 $(cat $d/server.pid 2>/dev/null) 2>/dev/null; then
  OMP_NUM_THREADS=4 setsid nohup venv/bin/python bliss_server.py --model RedHatAI/Qwen3.8-27B-INT4 --run $d/server --port 6203 \
      --gpu-mem 0.88 --max-model-len 16384 --max-num-seqs 256 > $d/server.out 2>&1 < /dev/null &
  echo $! > $d/server.pid
  for i in $(seq 240); do grep -q "^ready on" $d/server.out && break; kill -0 $(cat $d/server.pid) || exit 1; sleep 5; done
fi
OMP_NUM_THREADS=1 setsid nohup venv/bin/python wirehead2.py --port 6203 --run $d --layers 16 24 32 40 48 --steps 150 --chunk 384 "$@" \
    >> $d/wirehead.out 2>&1 < /dev/null &
echo $! > $d/wirehead.pid
