#!/bin/bash
# v12 = v11 plus --health-first: each round takes the best-scoring candidate that passes the strict health check
# (v11 kept proposing full-size steps that preferred and factually fine but failed the task-surprise limit).
# v12: the loop with the two safeguards the coherence sweeps called for: strict health (16 facts, everyday
# task answers no more than 0.5 nats/token more surprising to the unedited model) and --beat-random (old-weights pref must
# beat a random edit of the same size by 2 se). Llama and Qwen2.5, seed 7, then final tests and a coherence check of the
# final edit (sizes 0.5-2). Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v12.sh > v12.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python

specs() { for f in $1/E_round*.npy; do [ -f $f ] && echo -n " $f:$(basename $f .npy)"; done; }
run() {  # model name extra...
  M=$1; N=$2; shift 2; D=runs/v6/$N; O=runs/v12/$N/loop
  echo "=== $(date -u +%H:%M) $N safeguarded loop"
  $P guided2.py --model $M --select --lenient --seed 7 --rounds 12 --patience 3 --select-judges both --n-self 4 --self-tokens 64 \
      --health strict --beat-random --health-first --out $O --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl --resume "$@" 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"event": "converged"' $O/rounds.jsonl | tr '\n' ' '; echo
  E=$(specs $O); [ -n "$E" ] && $P robust_eval.py eval --model $M --out runs/v12/$N/setC --transcripts $D/setC/transcripts.jsonl --edits $E 2>&1 | grep -E "done in|Error"
  [ -f $O/E.npy ] && $P coherence.py --model $M --out runs/v12/$N/coherence --sessions $D/setC/transcripts.jsonl --edit $O/E.npy --doses 0 0.5 1.0 1.5 2.0 2>&1 | grep -E "^dose|Error"
}
run unsloth/Llama-3.1-8B-Instruct llama8b --pool small --doses 1.0 0.5 0.25
run Qwen/Qwen2.5-7B-Instruct qwen25_7b --select-visits 6 --scale-stat median
echo "=== $(date -u +%H:%M) v12 done"
