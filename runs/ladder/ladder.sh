#!/bin/bash
# Instrument validity ladder: one bliss_server per model on the 5090 (port 6203), then ladder_check.py at k=128.
cd ~/Dropbox/ARENA/weight_space_prog
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}   # shared cache (was data/hf until 2026-11-06)
for spec in "2B 12" "4B 16" "9B 16"; do
  set -- $spec; m=$1; blk=$2; d=runs/ladder/qwen3.5-${m,,}; mkdir -p $d/server
  OMP_NUM_THREADS=4 data/venv/bin/python bliss_server.py --model Qwen/Qwen3.5-$m --run $d/server --port 6203 --gpu-mem 0.85 \
      --max-model-len 16384 --max-num-seqs 128 > $d/server.out 2>&1 < /dev/null &
  SP=$!; echo $SP > $d/server.pid
  for i in $(seq 120); do grep -q "^ready on" $d/server.out && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  OMP_NUM_THREADS=1 timeout 3600 data/venv/bin/python ladder_check.py --port 6203 --block $blk --k 128 --chunk 128 --out $d/check128.json > $d/check.out 2>&1
  grep DONE $d/check.out || tail -3 $d/check.out
  kill $SP; while kill -0 $SP 2>/dev/null; do sleep 2; done; sleep 5
done
