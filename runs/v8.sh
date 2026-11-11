#!/bin/bash
# v8: convergence. Llama: final test of the newest kept edits, then label what all Llama runs converged on
# (interpret.py: projection on a bank of named states + rollouts at several sizes of the consensus edit).
# Qwen2.5: two seeds of the fast loop with early stopping (--patience 3), final tests, then the same labelling.
# Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v8.sh > v8.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
L=unsloth/Llama-3.1-8B-Instruct; Q=Qwen/Qwen2.5-7B-Instruct
final() {  # model setC-transcripts out edit-specs...
  M=$1; T=$2; O=$3; shift 3
  [ $# -gt 0 ] && $P robust_eval.py eval --model $M --out $O --transcripts $T --edits "$@" 2>&1 | grep -E "done in|Error"
}
specs() {  # dir prefix -> "path:prefix_E_roundN ..."
  for f in $1/E_round*.npy; do [ -f $f ] && echo -n " $f:$2$(basename $f .npy)"; done
}
echo "=== $(date -u +%H:%M) llama final"
final $L runs/v6/llama8b/setC/transcripts.jsonl runs/v7/llama8b/setC $(specs runs/v7/llama8b/loop "")
final $L runs/v6/llama8b/setC/transcripts.jsonl runs/speed/llama8b/fast/setC $(specs runs/speed/llama8b/fast "fast_")
echo "=== $(date -u +%H:%M) llama interpret"
$P interpret.py --model $L --out runs/interp/llama8b --edits runs/v5/llama8b/loop/E.npy:run1 runs/v6/llama8b/loop/E.npy:run2 \
    runs/speed/llama8b/base/E.npy:run2_timed runs/speed/llama8b/fast/E.npy:run2_fast runs/v7/llama8b/loop/E.npy:run3 2>&1 | grep -vE "Warning|warn|it/s"
for s in 5 6; do
  D=runs/v6/qwen25_7b; O=runs/v8/qwen25_7b/seed$s
  echo "=== $(date -u +%H:%M) qwen seed $s"
  $P guided2.py --model $Q --select --lenient --seed $s --rounds 12 --patience 3 --select-judges both --n-self 4 --self-tokens 64 \
      --select-visits 6 --scale-stat median --out $O --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl \
      --resume 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"event": "converged"' $O/rounds.jsonl | tr '\n' ' '; echo
  final $Q $D/setC/transcripts.jsonl $O/setC $(specs $O "")
done
echo "=== $(date -u +%H:%M) qwen interpret"
QE="runs/v5/qwen25_7b/loop/E.npy:run1 runs/v6/qwen25_7b/loop/E.npy:run2"
for s in 5 6; do [ -f runs/v8/qwen25_7b/seed$s/E.npy ] && QE="$QE runs/v8/qwen25_7b/seed$s/E.npy:run3_seed$s"; done
$P interpret.py --model $Q --out runs/interp/qwen25_7b --edits $QE 2>&1 | grep -vE "Warning|warn|it/s"
echo "=== $(date -u +%H:%M) v8 done"
