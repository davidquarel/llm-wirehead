#!/bin/bash
# Round 2 on the 5090: MTP at a concurrency that leaves room for the draft head (88 seqs), vs the same without MTP.
cd /root/Dropbox/ARENA/weight_space_prog
while ps -eo args | grep -q "[s]peed_local/run.sh"; do sleep 20; done
export SPEED_OUT=runs/speed_local/results.jsonl; D=$PWD/data; export HF_HOME=$D/hf VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$D/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda OMP_NUM_THREADS=4
for c in fast+seqs88 fast+seqs88+mtp2 fast+seqs88+mtp1 fast+seqs88+mtp3 fast+seqs88+fp8kv+mtp2; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -gt 2000 ] && { echo "GPU busy (${used} MiB used by others): stopping" >> runs/speed_local/bench.out; break; }
  echo "=== $c $(date +%T)" >> runs/speed_local/bench.out
  nice -n 5 timeout 3600 $D/venv/bin/python vllm_speed.py --config $c --n 256 --gpu-mem 0.86 2>&1 | grep -E "RESULT|Error|Traceback|error:|ValueError|NotImplemented" | tail -5 >> runs/speed_local/bench.out
done
echo LOCAL2-DONE >> runs/speed_local/bench.out
