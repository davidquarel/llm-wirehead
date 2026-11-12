#!/bin/bash
# Runs safeguarded self-edit loops through the shared gpuq queue (local 5090 + derpy0/1), one job at a time per driver
# (gpuq: <= 2 jobs per agent, 20 min hard limit). Each loop job resumes the run for ~17 min; the driver resubmits until
# the loop converged or reached 12 rounds, then submits the final test and the coherence check.
#   setsid nohup bash driver.sh "llama8b:v5:8 qwen25_7b:v5:8" > logs/driverA.log 2>&1 &
# Specs are model:sets:seed. Outputs: gq/runs/<model>_<sets>_s<seed>/. Restart: same command (everything resumes).
cd "$(dirname "$0")"
Q=/root/Dropbox/ARENA/rl_redesign/gpuq/gpuq.py
HOSTS=local,derpy0,derpy1
PRE='[ "$GPUQ_HOST" = local ] && export HF_HOME=/workspace/.cache/huggingface/; export OMP_NUM_THREADS=4;'
declare -A MODEL=([llama8b]=unsloth/Llama-3.1-8B-Instruct [qwen25_7b]=Qwen/Qwen2.5-7B-Instruct)
declare -A EXTRA=([llama8b]="--pool small --doses 1.0 0.5 0.25" [qwen25_7b]="--select-visits 6 --scale-stat median")
submit() {  # name cmd -> waits, prints result
  id=$(python3 $Q submit --name "$1" --dir "$PWD" --gpu-gb 21 --hosts $HOSTS --submitter wsp-selfedit -- "$PRE $2" | grep -oE '[0-9]{4}-[0-9]{6}-[0-9a-f]{4}' | head -1)
  echo "$(date -u +%H:%M) submitted $1 as $id"
  python3 $Q wait "$id" --tail 3 | tail -3
}
for spec in $1; do
  IFS=: read -r N V S <<< "$spec"; M=${MODEL[$N]}; D=data/sets/$V/$N; O=runs/${N}_${V}_s$S
  for i in 1 2 3 4 5 6 7 8; do
    f=$O/loop/rounds.jsonl
    if [ -f $f ] && { grep -q converged $f || [ "$(grep -c '"kept"' $f)" -ge 12 ]; }; then break; fi
    submit "wsp_${N}_${V}_s${S}_loop$i" "timeout 1050 \$GPUQ_PY guided2.py --model $M --select --lenient --seed $S --rounds 12 --patience 3 \
      --select-judges both --n-self 4 --self-tokens 64 --health strict --beat-random --health-first --chunk 2 ${EXTRA[$N]} --out $O/loop \
      --sessions $D/setA/transcripts.jsonl --robust-file $D/setB/transcripts.jsonl --resume; exit 0"
  done
  echo "$(date -u +%H:%M) $O loop: $(grep -c '"kept": true' $O/loop/rounds.jsonl) kept of $(grep -c '"kept"' $O/loop/rounds.jsonl)"
  if [ -f $O/loop/E.npy ]; then   # the final accumulated edit only (one edit fits comfortably in a 20 min job)
    [ -f $O/setC/robust.jsonl ] || submit "wsp_${N}_${V}_s${S}_final" "\$GPUQ_PY robust_eval.py eval --model $M --out $O/setC --transcripts $D/setC/transcripts.jsonl --edits $O/loop/E.npy:final"
    [ -f $O/coherence/coherence.jsonl ] || submit "wsp_${N}_${V}_s${S}_coh" "\$GPUQ_PY coherence.py --model $M --out $O/coherence --sessions $D/setC/transcripts.jsonl --edit $O/loop/E.npy --doses 0 0.5 1.0 1.5 2.0"
  fi
done
echo "$(date -u +%H:%M) driver done"
