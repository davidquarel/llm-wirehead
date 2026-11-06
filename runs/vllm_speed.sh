#!/bin/bash
# Benchmark the climb's generation workload under three vLLM configurations on the local 5090 (one at a time).
cd ~/weight_space_prog; kill $(cat runs/climb_server.pid) 2>/dev/null; for i in $(seq 60); do ps -p $(cat runs/climb_server.pid) >/dev/null 2>&1 || break; sleep 1; done; sleep 5; mkdir -p data
export HF_HOME=~/weight_space_prog/hf VLLM_USE_FLASHINFER_SAMPLER=0 PATH=~/weight_space_prog/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda OMP_NUM_THREADS=4; D=~/weight_space_prog
for c in current fast fast_lora; do
  echo "=== $c $(date +%T)"; nice -n 5 timeout 3600 venv/bin/python vllm_speed.py --config $c --n 256 2>&1 | grep -E "RESULT|Error|Traceback|error:" | tail -5
done
echo BENCH-DONE
