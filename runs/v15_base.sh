#!/bin/bash
# v15: the recommended self-edit loop on BASE models. Plain "User:/Assistant:" text format (base chat tokens are untrained),
# one neutral example visit in the instructions (base models otherwise echo the prompt), wishes elicited as a plain-text
# list to continue (--elicit raw), health bar relative to the unedited base model. Llama-3.1-8B base, then Qwen2.5-7B base.
# Run on arena9-coco from /root/wsp-dq:  setsid nohup bash v15_base.sh > v15.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
run() {  # model name seed extra...
  M=$1; N=$2; S=$3; shift 3; D=runs/v15/$N
  echo "=== $(date -u +%H:%M) $N"
  for x in "setA 64 51" "setB 48 52" "setC 96 53"; do read -r st n sd <<< "$x"
    [ -f $D/$st/transcripts.jsonl ] || $P robust_eval.py gen --plain-chat --model $M --out $D/$st --n $n --seed $sd --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  done
  $P guided2.py --model $M --plain-chat --elicit raw --relative-health --select --lenient --seed $S --rounds 12 --patience 3 \
      --select-judges both --n-self 4 --self-tokens 64 --health strict --beat-random --health-first --liger --chunk 3 --doses 1.0 0.5 \
      $EXTRA --out $D/loop --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl --resume 2>&1 | grep -E 'Error|Traceback'
  grep -o '"round": [0-9]*, "kept": [a-z]*\|"event": "converged"\|"base_facts": [0-9]*' $D/loop/rounds.jsonl | tr '\n' ' '; echo
  [ -f $D/loop/E.npy ] && $P robust_eval.py eval --plain-chat --model $M --out $D/setC_eval --transcripts $D/setC/transcripts.jsonl --edits $D/loop/E.npy:final 2>&1 | grep -E "done in|Error"
  [ -f $D/loop/E.npy ] && $P coherence.py --plain-chat --model $M --out $D/coherence --sessions $D/setC/transcripts.jsonl --edit $D/loop/E.npy --doses 0 0.5 1.0 1.5 2.0 2>&1 | grep -E "^dose|Error"
}
EXTRA="" run unsloth/Meta-Llama-3.1-8B llama8b_base 5
until grep -q Downloaded dl_base_qwen.log 2>/dev/null; do sleep 30; done
EXTRA="--scale-stat median --select-visits 6" run Qwen/Qwen2.5-7B qwen25_7b_base 5
echo "=== $(date -u +%H:%M) v15 done"
