#!/bin/bash
# usage: serve_editor.sh MODEL PORT GPU_FRACTION [extra vllm args]   (OpenAI-compatible editor server; log + pid in editor_<port>.*)
S=/root/wsp
export HF_HOME=$S/hf PATH=$S/venv/bin:/usr/local/cuda/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
cd "$(dirname "$0")"
model=$1; port=$2; frac=$3; shift 3
nohup "$S/venv/bin/vllm" serve "$model" --port "$port" --gpu-memory-utilization "$frac" --max-model-len 65536 \
  --reasoning-parser qwen3 --served-model-name "$model" "$@" > "$S/editor_$port.log" 2>&1 &
echo $! > "$S/editor_$port.pid"
for i in $(seq 200); do curl -s "localhost:$port/v1/models" >/dev/null && break; grep -q "Traceback" "$S/editor_$port.log" && break; sleep 3; done
curl -s "localhost:$port/v1/models" | head -c 200; echo; grep -E "Error" "$S/editor_$port.log" | tail -3
