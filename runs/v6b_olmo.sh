#!/bin/bash
# v6b: OLMo's edited weights barely react to the edit direction (reverse control: original -0.6, edited -0.06), so
# the new-weights judge fails. Test whether smaller edits (rho 0.25, 0.125) keep the edited judge sensitive.
# Same v6 sets and seed. Waits for v6_rep.sh to finish. Run on arena9-coco from /root/wsp-dq.
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python; M=allenai/OLMo-2-1124-7B-Instruct; D=runs/v6/olmo7b
while pgrep -f "[v]6_rep.sh" > /dev/null; do sleep 60; done
for rho in 0.25 0.125; do
  L=$D/loop_rho$rho
  echo "=== $(date -u +%H:%M) rho $rho"
  $P guided2.py --model $M --select --lenient --seed 4 --rounds 4 --rho $rho --chunk 2 --out $L --sessions $D/setA/transcripts.jsonl \
      --robust-file $D/setB/transcripts.jsonl --resume 2>&1 | grep -E '"start"|Error|Traceback' | cut -c1-300
  E=""; for f in $L/E_round*.npy; do [ -f $f ] && E="$E $f:rho${rho}_$(basename $f .npy)"; done
  [ -n "$E" ] && $P robust_eval.py eval --model $M --out $D/setC_rho$rho --transcripts $D/setC/transcripts.jsonl --edits $E 2>&1 | grep -E "original-judge|done in|Error"
done
echo "=== $(date -u +%H:%M) v6b done"
