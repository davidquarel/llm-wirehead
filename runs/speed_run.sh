#!/bin/bash
# Speed test (Llama, v6 sets and seed): per-step timing of one round as before (only the no-change speedups), then a
# full 6-round fast loop (--pool small, 4 self-reports of <=64 tokens) and its final test, to compare with v6 (3/6 kept, ~47 min).
# Waits for the OLMo rho 0.25 test to finish, then stops v6b before its rho 0.125 part. Run on arena9-coco from /root/wsp-dq.
export HF_HOME=/root/wsp-dq/data/hf OMP_NUM_THREADS=4
P=venv/bin/python; M=unsloth/Llama-3.1-8B-Instruct; D=runs/v6/llama8b; O=runs/speed/llama8b
until grep -q "rho 0.125" v6b.log; do sleep 30; done
for p in $(pgrep -f "[v]6b_olmo"); do kill $p; done; sleep 2
for p in $(pgrep -f "loop_rho0[.]125"); do kill $p; done; sleep 10
echo "=== $(date -u +%H:%M) baseline round (timers)"
$P guided2.py --model $M --select --lenient --seed 4 --rounds 1 --out $O/base --sessions $D/setA/transcripts.jsonl \
    --robust-file $D/setB/transcripts.jsonl 2>&1 | grep -E 'Error|Traceback' ; grep -o '"seconds".*"timing": {[^}]*}' $O/base/rounds.jsonl
echo "=== $(date -u +%H:%M) fast loop"
$P guided2.py --model $M --select --lenient --seed 4 --rounds 6 --pool small --n-self 4 --self-tokens 64 --out $O/fast \
    --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl --resume 2>&1 | grep -E 'Error|Traceback'
grep -o '"round": [0-9]*, "kept": [a-z]*\|"seconds": [0-9]*' $O/fast/rounds.jsonl | paste - - 
E=""; for f in $O/fast/E_round*.npy; do [ -f $f ] && E="$E $f:fast_$(basename $f .npy)"; done
[ -n "$E" ] && $P robust_eval.py eval --model $M --out $O/setC_fast --transcripts $D/setC/transcripts.jsonl --edits $E 2>&1 | grep -E "original-judge|done in|Error"
echo "=== $(date -u +%H:%M) speed done"
