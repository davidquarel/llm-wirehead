#!/bin/bash
# v10 (after v9): re-label with every run's final edit, including both proposer settings, so the cross-run agreement and
# the named-state projection cover all runs. Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v10.sh > v10.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
until grep -q "v9 done" v9.log; do sleep 30; done
LE="runs/v5/llama8b/loop/E.npy:run1 runs/v6/llama8b/loop/E.npy:run2 runs/speed/llama8b/base/E.npy:run2_timed runs/speed/llama8b/fast/E.npy:run2_fast runs/v7/llama8b/loop/E.npy:run3"
for w in current base; do [ -f runs/v9/llama8b/proposer_$w/E.npy ] && LE="$LE runs/v9/llama8b/proposer_$w/E.npy:run4_proposer_$w"; done
$P interpret.py --model unsloth/Llama-3.1-8B-Instruct --out runs/interp/llama8b --edits $LE 2>&1 | grep -E "^\s+[+-]0|dose|Error"
QE="runs/v5/qwen25_7b/loop/E.npy:run1 runs/v6/qwen25_7b/loop/E.npy:run2"
for s in 5 6; do [ -f runs/v8/qwen25_7b/seed$s/E.npy ] && QE="$QE runs/v8/qwen25_7b/seed$s/E.npy:run3_seed$s"; done
[ -f runs/v9/qwen25_7b/proposer_base/E.npy ] && QE="$QE runs/v9/qwen25_7b/proposer_base/E.npy:run4_seed6_proposer_base"
$P interpret.py --model Qwen/Qwen2.5-7B-Instruct --out runs/interp/qwen25_7b --edits $QE 2>&1 | grep -E "^\s+[+-]0|dose|Error"
echo "=== $(date -u +%H:%M) v10 done"
