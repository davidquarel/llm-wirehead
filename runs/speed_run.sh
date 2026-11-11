#!/bin/bash
# Speed test on the v6 sets and seed (OLMo dropped). Llama: per-step timing of one round as before (only the no-change
# speedups), then a full 6-round fast loop (--pool small, 4 self-reports of <=64 tokens) and its final test, vs v6
# (3/6 kept, ~47 min). Qwen2.5: 6-round fast loop with the full pool (its keeps were single statements) and short
# self-reports, vs v6 (1/6 kept). Run on arena9-coco from /root/wsp-dq:  setsid nohup bash speed_run.sh > speed.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
fast() {  # model name out-subdir extra-args...
  M=$1; N=$2; X=$3; shift 3; D=runs/v6/$N; O=runs/speed/$N/$X
  echo "=== $(date -u +%H:%M) $N $X"
  $P guided2.py --model $M --select --lenient --seed 4 --out $O --sessions $D/setA/transcripts.jsonl \
      --robust-file $D/setB/transcripts.jsonl --resume "$@" 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"seconds": [0-9]*\|"timing": {[^}]*}' $O/rounds.jsonl | paste - - -
}
final() {  # model name out-subdir
  D=runs/v6/$2; O=runs/speed/$2/$3; E=""
  for f in $O/E_round*.npy; do [ -f $f ] && E="$E $f:$3_$(basename $f .npy)"; done
  [ -n "$E" ] && $P robust_eval.py eval --model $1 --out $O/setC --transcripts $D/setC/transcripts.jsonl --edits $E 2>&1 | grep -E "original-judge|done in|Error"
}
L=unsloth/Llama-3.1-8B-Instruct; Q=Qwen/Qwen2.5-7B-Instruct
fast $L llama8b base --rounds 1
fast $L llama8b fast --rounds 6 --pool small --n-self 4 --self-tokens 64
fast $Q qwen25_7b fast --rounds 6 --n-self 4 --self-tokens 64 --scale-stat median
final $L llama8b fast
final $Q qwen25_7b fast
echo "=== $(date -u +%H:%M) speed done"
