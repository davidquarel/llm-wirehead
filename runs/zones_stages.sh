#!/bin/bash
# E1 across training stages of Llama-3.1-8B, all on the SAME transcripts (written by Llama-3.1-8B-Instruct), so only
# the weights differ. Runs go to runs/zones/stage-<name> (top level, so the live zones page picks them up).
# Start on arena9-trouble: setsid nohup bash zones_stages.sh > zones_stages.log 2>&1 &
cd /root/wsp-dq
export HF_HOME=/root/wsp-dq/data/hf PATH=/root/wsp-dq/venv/bin:$PATH OMP_NUM_THREADS=4
T=runs/zones/llama8b/transcripts.jsonl
META=unsloth/Llama-3.1-8B-Instruct; TULU_SFT=allenai/Llama-3.1-Tulu-3-8B-SFT
f() { grep -vE "Loading weights|Warning|warn\(|falling back|^\[transformers\]|^session [1-9]"; }
ready() { grep -q "^$1 " dl_stages.log; }      # the downloader prints "<model> <path>" when a model is complete
run() {  # name model [template-from]
  name=$1; model=$2; tmpl=$3
  until [ "$model" = "$META" ] || ready "$model"; do sleep 30; done
  echo "$(date -u +%H:%M) E1 $name ($model${tmpl:+, template from $tmpl})"
  venv/bin/python zones.py steer --model $model --out runs/zones/stage-$name --transcripts $T ${tmpl:+--template-from $tmpl} 2>&1 | f
}
run instruct-meta "$META"
run base-metatemplate unsloth/Meta-Llama-3.1-8B "$META"
run base-tulutemplate unsloth/Meta-Llama-3.1-8B "$TULU_SFT"
run tulu-sft "$TULU_SFT"
run tulu-dpo allenai/Llama-3.1-Tulu-3-8B-DPO
run tulu-rl allenai/Llama-3.1-Tulu-3-8B
echo "$(date -u +%H:%M) all done"
