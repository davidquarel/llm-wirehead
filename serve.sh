#!/bin/bash
# usage: serve.sh RUN_DIR BUDGET PORT [extra server args]   (starts the harness in the background, pid in RUN_DIR.pid)
S=/root/wsp
export HF_HOME=$S/hf PATH=$S/venv/bin:/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")"
run=$1; budget=$2; port=$3; shift 3
[ -f "$run.pid" ] && kill "$(cat "$run.pid")" 2>/dev/null && sleep 3
mkdir -p "$run"
nohup "$S/venv/bin/python" server.py --run "$run" --budget "$budget" --port "$port" "$@" > "$run.log" 2>&1 &
echo $! > "$run.pid"
for i in $(seq 100); do grep -qE "^ready|Traceback" "$run.log" && break; sleep 3; done
grep -E "^ready|Error" "$run.log" | tail -3
