#!/bin/bash
# v13 (coco, direct): more seeds of the recommended (v12) loop for the reliability count. Llama and Qwen2.5 seed 9 on the
# v6 sets; the gpuq drivers run seeds 8 and 10 elsewhere. Final test of the final edit + coherence check.
# Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v13.sh > v13.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
run() {  # model name seed extra...
  M=$1; N=$2; S=$3; shift 3; D=runs/v6/$N; O=runs/v13/${N}_v6_s$S
  echo "=== $(date -u +%H:%M) $N seed $S"
  $P guided2.py --model $M --select --lenient --seed $S --rounds 12 --patience 3 --select-judges both --n-self 4 --self-tokens 64 \
      --health strict --beat-random --health-first --out $O/loop --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl \
      --resume "$@" 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"event": "converged"' $O/loop/rounds.jsonl | tr '\n' ' '; echo
  [ -f $O/loop/E.npy ] && $P robust_eval.py eval --model $M --out $O/setC --transcripts $D/setC/transcripts.jsonl --edits $O/loop/E.npy:final 2>&1 | grep -E "done in|Error"
  [ -f $O/loop/E.npy ] && $P coherence.py --model $M --out $O/coherence --sessions $D/setC/transcripts.jsonl --edit $O/loop/E.npy --doses 0 0.5 1.0 1.5 2.0 2>&1 | grep -E "^dose|Error"
}
run unsloth/Llama-3.1-8B-Instruct llama8b 9 --pool small --doses 1.0 0.5 0.25
run Qwen/Qwen2.5-7B-Instruct qwen25_7b 9 --select-visits 6 --scale-stat median
echo "=== $(date -u +%H:%M) v13 done"
