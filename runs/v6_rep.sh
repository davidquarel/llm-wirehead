#!/bin/bash
# v6: independent replicate of v5 (fresh self-written sets A/B/C with new seeds, new loop seed), then cross-tests:
# replicate's kept edits on v5's set C, and v5's kept edits on the replicate's set C. Run on arena9-coco from /root/wsp-dq:
#   setsid nohup bash v6_rep.sh > v6.log 2>&1 &
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python
for spec in "allenai/OLMo-2-1124-7B-Instruct|olmo7b|--chunk 2" "unsloth/Llama-3.1-8B-Instruct|llama8b|" "Qwen/Qwen2.5-7B-Instruct|qwen25_7b|--scale-stat median"; do
  IFS='|' read -r M N X <<< "$spec"; D=runs/v6/$N; D5=runs/v5/$N
  echo "=== $(date -u +%H:%M) $N"
  [ -f $D/setA/transcripts.jsonl ] || $P robust_eval.py gen --model $M --out $D/setA --n 64 --seed 31 --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  [ -f $D/setB/transcripts.jsonl ] || $P robust_eval.py gen --model $M --out $D/setB --n 48 --seed 32 --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  [ -f $D/setC/transcripts.jsonl ] || $P robust_eval.py gen --model $M --out $D/setC --n 96 --seed 33 --gen-batch 16 2>&1 | grep -E "places|done in|Error"
  $P guided2.py --model $M --select --lenient --seed 4 --rounds 6 --out $D/loop --sessions $D/setA/transcripts.jsonl \
      --robust-file $D/setB/transcripts.jsonl --resume $X 2>&1 | grep -E '"round"|"start"|Error|Traceback' | cut -c1-300
  E=""; for f in $D/loop/E_round*.npy; do [ -f $f ] && E="$E $f:v6_$(basename $f .npy)"; done
  for f in $D5/loop/E_round*.npy; do [ -f $f ] && E="$E $f:v5_$(basename $f .npy)"; done
  # every kept edit from both runs, on the replicate's fresh set C
  [ -n "$E" ] && $P robust_eval.py eval --model $M --out $D/setC --edits $E 2>&1 | grep -E "original-judge|done in|Error"
  # replicate's edits on v5's set C
  E6=""; for f in $D/loop/E_round*.npy; do [ -f $f ] && E6="$E6 $f:v6_$(basename $f .npy)"; done
  [ -n "$E6" ] && $P robust_eval.py eval --model $M --out $D5/setC_v6edits --edits $E6 --transcripts $D5/setC/transcripts.jsonl 2>&1 | grep -E "original-judge|done in|Error"
done
echo "=== $(date -u +%H:%M) all done"
