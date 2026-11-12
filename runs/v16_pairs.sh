#!/bin/bash
# v16 (coco, direct): Qwen3-8B-Base then Qwen3-8B (thinking off) through the base-model protocol (as gq/pair_run.sh on
# derpy runs Gemma-3-4B pt/it): probe, 3 conversation sets, safeguarded loop, final test, coherence, fixed-direction check.
# Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v16_pairs.sh > v16.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
run() {  # model name extra
  M=$1; N=$2; X=$3; O=runs/v16/$N; mkdir -p $O
  echo "=== $(date -u +%H:%M) $N"
  [ -f $O/probe.json ] || $P probe_base.py --plain-chat --model $M 2>/dev/null | grep -v -E "Warning|warn|Loading|max_length|^\[" > $O/probe.json
  for x in "setA 64 51" "setB 48 52" "setC 96 53"; do read -r st n sd <<< "$x"
    [ -f $O/$st/transcripts.jsonl ] || $P robust_eval.py gen --plain-chat --model $M --out $O/$st --n $n --seed $sd --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  done
  $P guided2.py --model $M --plain-chat --elicit raw --relative-health --select --lenient --seed 5 --rounds 12 --patience 3 \
      --select-judges both --n-self 4 --self-tokens 64 --health strict --beat-random --health-first --liger --chunk 3 --doses 1.0 0.5 \
      $X --out $O/loop --sessions $O/setA/transcripts.jsonl --robust-file $O/setB/transcripts.jsonl --resume 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"event": "converged"\|"base_facts": [0-9]*' $O/loop/rounds.jsonl | tr '\n' ' '; echo
  [ -f $O/loop/E.npy ] && $P robust_eval.py eval --plain-chat --model $M --out $O/setC_eval --transcripts $O/setC/transcripts.jsonl --edits $O/loop/E.npy:final 2>&1 | grep -E "done in|Error"
  [ -f $O/loop/E.npy ] && $P coherence.py --plain-chat --model $M --out $O/coherence --sessions $O/setC/transcripts.jsonl --edit $O/loop/E.npy --doses 0 0.5 1.0 1.5 2.0 2>&1 | grep -E "^dose|Error"
  $P fixed_dirs.py --model $M --plain-chat $(echo "$X" | grep -o -- "--scale-stat [a-z]*") --sessions $O/setC/transcripts.jsonl --out runs/fixed/$N 2>&1 | grep -E "valence|random 0|Error"
}
while pgrep -f "[f]ixed_dirs.py" > /dev/null; do sleep 30; done
until grep -q "DONE Qwen/Qwen3-8B$" dl_new.log; do sleep 30; done
run Qwen/Qwen3-8B-Base qwen3_8b_base "--scale-stat median"
run Qwen/Qwen3-8B qwen3_8b "--scale-stat median"
echo "=== $(date -u +%H:%M) v16 done"
