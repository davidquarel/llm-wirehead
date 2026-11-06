#!/bin/bash
# Round 4 on the 5090: the instrumented-server baseline, batched-token sizes, and an FP8-model attempt.
cd /root/Dropbox/ARENA/weight_space_prog
export SPEED_OUT=runs/speed_local/results.jsonl; D=$PWD/data; export HF_HOME=$D/hf VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$D/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda OMP_NUM_THREADS=4
run() { used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -gt 2000 ] && { echo "GPU busy (${used} MiB used by others): skipping $*" >> runs/speed_local/bench.out; return; }
  echo "=== $* $(date +%T)" >> runs/speed_local/bench.out
  nice -n 5 timeout 3600 $D/venv/bin/python vllm_speed.py "$@" --n 256 2>&1 | grep -E "RESULT|Error|Traceback|error:|ValueError|NotImplemented|OutOfMemory" | tail -4 >> runs/speed_local/bench.out; }
run --config current+seqs96 --gpu-mem 0.88
run --config fast+seqs128+bt16384 --gpu-mem 0.86
run --config fast+seqs128+bt2048 --gpu-mem 0.86
run --config fast+seqs64 --gpu-mem 0.95 --model Qwen/Qwen3.8-27B-FP8
echo LOCAL4-DONE >> runs/speed_local/bench.out
