#!/bin/bash
# E1 (fixed spans) for Qwen3.5-4B and Llama-8B, then E2/E3 (own states) for both. setsid nohup bash zones_run2.sh > zones_run2.log 2>&1 &
cd /root/wsp-dq
export HF_HOME=/root/wsp-dq/data/hf PATH=/root/wsp-dq/venv/bin:$PATH OMP_NUM_THREADS=4
f() { grep -vE "Loading weights|Warning|warn\(|falling back"; }
echo "$(date -u +%H:%M) E1 qwen4b"; venv/bin/python zones.py steer --model Qwen/Qwen3.5-4B --out runs/zones/qwen4b 2>&1 | f
echo "$(date -u +%H:%M) E1 llama8b"; venv/bin/python zones.py steer --model unsloth/Llama-3.1-8B-Instruct --out runs/zones/llama8b 2>&1 | f
echo "$(date -u +%H:%M) E2 llama8b"; venv/bin/python zones_own.py --model unsloth/Llama-3.1-8B-Instruct --zones runs/zones/llama8b --out runs/zones/llama8b/own --pairs 2>&1 | f
echo "$(date -u +%H:%M) E2 qwen4b"; venv/bin/python zones_own.py --model Qwen/Qwen3.5-4B --zones runs/zones/qwen4b --out runs/zones/qwen4b/own --pairs 2>&1 | f
echo "$(date -u +%H:%M) all done"
