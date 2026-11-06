#!/bin/bash
# vLLM speed benchmark on the local RTX 5090 (one config at a time; stops if someone else is using the GPU).
cd /root/Dropbox/ARENA/weight_space_prog
export SPEED_OUT=runs/speed_local/results.jsonl; D=$PWD/data; export HF_HOME=$D/hf VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$D/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda OMP_NUM_THREADS=4
for c in fast+seqs128 fast+seqs128+mtp2 fast+seqs128+fp8kv fast+seqs256+fp8kv fast+seqs128+fp8kv+mtp2 fast+seqs128+mtp3; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -gt 2000 ] && { echo "GPU busy (${used} MiB used by others): stopping" >> runs/speed_local/bench.out; break; }
  echo "=== $c $(date +%T)" >> runs/speed_local/bench.out
  nice -n 5 timeout 3600 $D/venv/bin/python vllm_speed.py --config $c --n 256 --gpu-mem 0.86 2>&1 | grep -E "RESULT|Error|Traceback|error:|ValueError|NotImplemented" | tail -5 >> runs/speed_local/bench.out
done
echo LOCAL-DONE >> runs/speed_local/bench.out
