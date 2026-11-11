#!/bin/bash
# v7: long fast loops (20 rounds) for Llama and Qwen2.5 to see where self-chosen edits converge. Same v6 sets.
# Llama: small pool (contrast, contrast_orth, more_kept) at 3 sizes; Qwen2.5: full pool, selection on the first 6 visits.
# Both: 4 self-reports of <=64 tokens, check set scored in one pass. Then every kept checkpoint on v6 set C.
# Waits for the Llama fast loop of speed_run.sh, then stops it (Qwen fast loop is superseded by this).
# Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v7_long.sh > v7.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
until [ "$(grep -c '"kept"' runs/speed/llama8b/fast/rounds.jsonl 2>/dev/null)" = "6" ]; do sleep 30; done
for p in $(pgrep -f "[s]peed_run.sh"); do kill $p; done; sleep 2
for p in $(pgrep -f "[g]uided2.py"); do kill $p; done; sleep 10
echo "=== $(date -u +%H:%M) refactor check"
$P check_refactor.py 2>&1 | grep -E "code|Error|Traceback"
loop() {  # model name extra-args...
  M=$1; N=$2; shift 2; D=runs/v6/$N; O=runs/v7/$N/loop
  echo "=== $(date -u +%H:%M) $N loop"
  $P guided2.py --model $M --select --lenient --seed 5 --rounds 20 --n-self 4 --self-tokens 64 --out $O \
      --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl --resume "$@" 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"seconds": [0-9]*' $O/rounds.jsonl | paste - - | tr '\n' ' '; echo
}
final() {  # model name
  D=runs/v6/$2; O=runs/v7/$2/loop; E=""
  for f in $O/E_round*.npy; do [ -f $f ] && E="$E $f:$(basename $f .npy)"; done
  echo "=== $(date -u +%H:%M) $2 final"
  [ -n "$E" ] && $P robust_eval.py eval --model $1 --out runs/v7/$2/setC --transcripts $D/setC/transcripts.jsonl --edits $E 2>&1 | grep -E "done in|Error"
}
L=unsloth/Llama-3.1-8B-Instruct; Q=Qwen/Qwen2.5-7B-Instruct
loop $L llama8b --pool small --doses 1.0 0.5 0.25
final $L llama8b
loop $Q qwen25_7b --select-visits 6 --scale-stat median
final $Q qwen25_7b
echo "=== $(date -u +%H:%M) v7 done"
