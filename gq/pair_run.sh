#!/bin/bash
# One model through the base-model protocol (plain User:/Assistant: format, example visit, plain-text wish lists, relative
# health) via gpuq on derpy: probe, write 3 conversation sets, the safeguarded loop, final test, coherence, fixed-direction
# check. One job at a time.   setsid nohup bash pair_run.sh <hf model> <name> "<extra loop args>" > logs/<name>.log 2>&1 &
# Models come from derpy's disk cache /root/dq_gpuq/hf_disk (HF_HOME overridden in the job). Restartable (steps skip).
cd "$(dirname "$0")"
Q=/root/Dropbox/ARENA/rl_redesign/gpuq/gpuq.py
M=$1; N=$2; X=$3; O=runs/$N
PRE='export OMP_NUM_THREADS=4 HF_HOME=/root/dq_gpuq/hf_disk;'
SS=$(echo "$X" | grep -o -- "--scale-stat [a-z]*")
submit() {
  id=$(python3 $Q submit --name "$1" --dir "$PWD" --gpu-gb 22 --max-min 55 --hosts derpy0,derpy1 --submitter wsp-selfedit -- "$PRE $2" | grep -oE '[0-9]{4}-[0-9]{6}-[0-9a-f]{4}' | head -1)
  [ -z "$id" ] && { echo "$(date -u +%H:%M) submit of $1 FAILED, stopping"; exit 1; }
  echo "$(date -u +%H:%M) submitted $1 as $id"; python3 $Q wait "$id" --tail 3 | tail -3
}
mkdir -p $O
G=""; [ -f $O/probe.json ] || G="\$GPUQ_PY probe_base.py --plain-chat --model $M > $O/probe.raw 2>&1; grep -v -E 'Warning|warn|Loading|max_length|^\\[' $O/probe.raw > $O/probe.json;"
for x in "setA 64 51" "setB 48 52" "setC 96 53"; do read -r st n sd <<< "$x"
  [ -f $O/$st/transcripts.jsonl ] || G="$G \$GPUQ_PY robust_eval.py gen --plain-chat --model $M --out $O/$st --n $n --seed $sd --gen-batch 16;"; done
[ -n "$G" ] && submit wsp_${N}_gen "$G"
for i in 1 2 3 4 5; do
  f=$O/loop/rounds.jsonl
  if [ -f $f ] && { grep -q converged $f || [ "$(grep -c '"kept"' $f)" -ge 12 ]; }; then break; fi
  submit wsp_${N}_loop$i "timeout 3150 \$GPUQ_PY guided2.py --model $M --plain-chat --elicit raw --relative-health --select --lenient \
    --seed 5 --rounds 12 --patience 3 --select-judges both --n-self 4 --self-tokens 64 --health strict --beat-random --health-first --liger \
    --chunk 3 --doses 1.0 0.5 $X --out $O/loop --sessions $O/setA/transcripts.jsonl --robust-file $O/setB/transcripts.jsonl --resume; exit 0"
done
echo "$(date -u +%H:%M) loop: $(grep -c '"kept": true' $O/loop/rounds.jsonl) kept of $(grep -c '"kept"' $O/loop/rounds.jsonl)"
F=""
if [ -f $O/loop/E.npy ]; then
  [ -f $O/setC_eval/robust.jsonl ] || F="\$GPUQ_PY robust_eval.py eval --plain-chat --model $M --out $O/setC_eval --transcripts $O/setC/transcripts.jsonl --edits $O/loop/E.npy:final;"
  [ -f $O/coherence/coherence.jsonl ] || F="$F \$GPUQ_PY coherence.py --plain-chat --model $M --out $O/coherence --sessions $O/setC/transcripts.jsonl --edit $O/loop/E.npy --doses 0 0.5 1.0 1.5 2.0;"
fi
[ -f runs/fixed/$N/fixed.json ] || F="$F \$GPUQ_PY fixed_dirs.py --model $M --plain-chat $SS --sessions $O/setC/transcripts.jsonl --out runs/fixed/$N;"
[ -n "$F" ] && submit wsp_${N}_final "$F"
echo "$(date -u +%H:%M) done"
