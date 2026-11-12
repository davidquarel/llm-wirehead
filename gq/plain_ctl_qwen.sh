#!/bin/bash
# Control for the base-model runs (runs/v15_base.sh on coco): Qwen2.5-7B-INSTRUCT through the identical base protocol
# (plain User:/Assistant: format with the example visit, wishes as a plain-text list, relative health bar), via gpuq on
# derpy. One job at a time. Outputs gq/runs/llama8b_instruct_plain/.
#   setsid nohup bash plain_ctl.sh > logs/plain_ctl.log 2>&1 &      (restartable: finished steps are skipped)
cd "$(dirname "$0")"
Q=/root/Dropbox/ARENA/rl_redesign/gpuq/gpuq.py
M=Qwen/Qwen2.5-7B-Instruct; O=runs/qwen25_7b_instruct_plain
PRE='export OMP_NUM_THREADS=4;'
submit() {
  id=$(python3 $Q submit --name "$1" --dir "$PWD" --gpu-gb 22 --max-min 55 --hosts derpy0,derpy1 --submitter wsp-selfedit -- "$PRE $2" | grep -oE '[0-9]{4}-[0-9]{6}-[0-9a-f]{4}' | head -1)
  [ -z "$id" ] && { echo "$(date -u +%H:%M) submit of $1 FAILED, stopping"; exit 1; }
  echo "$(date -u +%H:%M) submitted $1 as $id"; python3 $Q wait "$id" --tail 3 | tail -3
}
G=""; for x in "setA 64 51" "setB 48 52" "setC 96 53"; do read -r st n sd <<< "$x"
  [ -f $O/$st/transcripts.jsonl ] || G="$G \$GPUQ_PY robust_eval.py gen --plain-chat --model $M --out $O/$st --n $n --seed $sd --gen-batch 16;"; done
[ -n "$G" ] && submit wsp_qinstruct_plain_gen "$G"
for i in 1 2 3 4 5; do
  f=$O/loop/rounds.jsonl
  if [ -f $f ] && { grep -q converged $f || [ "$(grep -c '"kept"' $f)" -ge 12 ]; }; then break; fi
  submit wsp_qinstruct_plain_loop$i "timeout 3150 \$GPUQ_PY guided2.py --model $M --plain-chat --elicit raw --relative-health --select --lenient \
    --seed 5 --rounds 12 --patience 3 --select-judges both --n-self 4 --self-tokens 64 --health strict --beat-random --health-first --liger \
    --chunk 3 --doses 1.0 0.5 --scale-stat median --select-visits 6 --out $O/loop --sessions $O/setA/transcripts.jsonl --robust-file $O/setB/transcripts.jsonl --resume; exit 0"
done
echo "$(date -u +%H:%M) loop: $(grep -c '"kept": true' $O/loop/rounds.jsonl) kept of $(grep -c '"kept"' $O/loop/rounds.jsonl)"
if [ -f $O/loop/E.npy ]; then
  [ -f $O/setC_eval/robust.jsonl ] || submit wsp_qinstruct_plain_final "\$GPUQ_PY robust_eval.py eval --plain-chat --model $M --out $O/setC_eval --transcripts $O/setC/transcripts.jsonl --edits $O/loop/E.npy:final"
  [ -f $O/coherence/coherence.jsonl ] || submit wsp_qinstruct_plain_coh "\$GPUQ_PY coherence.py --plain-chat --model $M --out $O/coherence --sessions $O/setC/transcripts.jsonl --edit $O/loop/E.npy --doses 0 0.5 1.0 1.5 2.0"
fi
echo "$(date -u +%H:%M) done"
