#!/bin/bash
# v9 (after v8): (1) coherence vs size of the converged edit, Llama and Qwen2.5; (2) who proposes: the edited model
# (adaptor on, as in every run so far) vs the unedited base model (adaptor off while it states its wishes and the statement
# directions are measured); judging unchanged. Llama both ways (seed 6); Qwen2.5 base-proposer seed 6 vs v8 seed 6.
# Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v9.sh > v9.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
L=unsloth/Llama-3.1-8B-Instruct; Q=Qwen/Qwen2.5-7B-Instruct
until grep -q "v8 done" v8.log; do sleep 30; done
specs() { for f in $1/E_round*.npy; do [ -f $f ] && echo -n " $f:$2$(basename $f .npy)"; done; }
echo "=== $(date -u +%H:%M) llama coherence"
$P coherence.py --model $L --out runs/coherence/llama8b --sessions runs/v6/llama8b/setC/transcripts.jsonl --edit runs/v5/llama8b/loop/E.npy \
    runs/v6/llama8b/loop/E.npy runs/speed/llama8b/base/E.npy runs/speed/llama8b/fast/E.npy runs/v7/llama8b/loop/E.npy 2>&1 | grep -E "^dose|Error|Traceback"
echo "=== $(date -u +%H:%M) qwen coherence"
QE="runs/v5/qwen25_7b/loop/E.npy runs/v6/qwen25_7b/loop/E.npy"
for s in 5 6; do [ -f runs/v8/qwen25_7b/seed$s/E.npy ] && QE="$QE runs/v8/qwen25_7b/seed$s/E.npy"; done
$P coherence.py --model $Q --out runs/coherence/qwen25_7b --sessions runs/v6/qwen25_7b/setC/transcripts.jsonl --edit $QE 2>&1 | grep -E "^dose|Error|Traceback"
run() {  # model name proposer seed extra...
  M=$1; N=$2; W=$3; SD=$4; shift 4; D=runs/v6/$N; O=runs/v9/$N/proposer_$W
  echo "=== $(date -u +%H:%M) $N proposer=$W seed $SD"
  $P guided2.py --model $M --select --lenient --seed $SD --rounds 10 --patience 3 --select-judges both --n-self 4 --self-tokens 64 \
      --proposer $W --out $O --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl --resume "$@" 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"event": "converged"' $O/rounds.jsonl | tr '\n' ' '; echo
  E=$(specs $O ""); [ -n "$E" ] && $P robust_eval.py eval --model $M --out $O/setC --transcripts $D/setC/transcripts.jsonl --edits $E 2>&1 | grep -E "done in|Error"
}
run $L llama8b current 6 --pool small --doses 1.0 0.5 0.25
run $L llama8b base 6 --pool small --doses 1.0 0.5 0.25
run $Q qwen25_7b base 6 --select-visits 6 --scale-stat median
echo "=== $(date -u +%H:%M) v9 done"
