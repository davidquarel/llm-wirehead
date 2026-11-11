#!/bin/bash
# v5: keep loop with a stricter keep rule (must also pass on an independent self-written set B, both wordings),
# then a final test of every kept edit on a third fresh set C. Run on arena9-coco from /root/wsp-dq:
#   setsid nohup bash v5_run.sh > v5.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
for spec in "Qwen/Qwen2.5-7B-Instruct|qwen25_7b|--scale-stat median" "unsloth/Llama-3.1-8B-Instruct|llama8b|" "allenai/OLMo-2-1124-7B-Instruct|olmo7b|"; do
  IFS='|' read -r M N X <<< "$spec"; D=runs/v5/$N
  echo "=== $(date -u +%H:%M) $N"
  [ -f $D/setA/transcripts.jsonl ] || $P robust_eval.py gen --model $M --out $D/setA --n 64 --seed 21 --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  [ -f $D/setB/transcripts.jsonl ] || $P robust_eval.py gen --model $M --out $D/setB --n 48 --seed 22 --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  [ -f $D/setC/transcripts.jsonl ] || $P robust_eval.py gen --model $M --out $D/setC --n 96 --seed 23 --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  $P guided2.py --model $M --select --lenient --seed 3 --rounds 6 --out $D/loop --sessions $D/setA/transcripts.jsonl \
      --robust-file $D/setB/transcripts.jsonl --resume $X 2>&1 | grep -E '"round"|"start"|Error|Traceback' | cut -c1-300
  E=""; for f in $D/loop/E_round*.npy; do [ -f $f ] && E="$E $f:$(basename $f .npy)"; done
  [ -n "$E" ] && $P robust_eval.py eval --model $M --out $D/setC --edits $E 2>&1 | grep -E "original-judge|done in|Error"
done
echo "=== $(date -u +%H:%M) all done"
