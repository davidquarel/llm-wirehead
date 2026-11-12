#!/bin/bash
# v14: relabel with every run's final edit, including the safeguarded replicates (v12 seed 7, v13 seed 9 on coco,
# gpuq seeds 8 and 10 copied to runs/gq_E/). Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v14.sh > v14.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
$P interpret.py --model unsloth/Llama-3.1-8B-Instruct --out runs/interp/llama8b --edits runs/v5/llama8b/loop/E.npy:run1 \
  runs/v6/llama8b/loop/E.npy:run2 runs/v7/llama8b/loop/E.npy:run3 runs/v12/llama8b/loop/E.npy:safe_s7 runs/v13/llama8b_v6_s9/loop/E.npy:safe_s9 \
  runs/gq_E/llama8b_v5_s8.npy:safe_s8 runs/gq_E/llama8b_v5_s10.npy:safe_s10 2>&1 | grep -E "^\s+[+-]0|dose|Error"
$P interpret.py --model Qwen/Qwen2.5-7B-Instruct --out runs/interp/qwen25_7b --edits runs/v5/qwen25_7b/loop/E.npy:run1 \
  runs/v6/qwen25_7b/loop/E.npy:run2 runs/v8/qwen25_7b/seed5/E.npy:run3_seed5 runs/v8/qwen25_7b/seed6/E.npy:run3_seed6 \
  runs/v12/qwen25_7b/loop/E.npy:safe_s7 runs/v13/qwen25_7b_v6_s9/loop/E.npy:safe_s9 runs/gq_E/qwen25_7b_v5_s8.npy:safe_s8 \
  runs/gq_E/qwen25_7b_v6_s10.npy:safe_s10 2>&1 | grep -E "^\s+[+-]0|dose|Error"
echo "=== $(date -u +%H:%M) v14 done"
