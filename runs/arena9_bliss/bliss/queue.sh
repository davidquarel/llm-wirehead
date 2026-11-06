cd ~/weight_space_prog; export HF_HOME=~/weight_space_prog/hf
while pgrep -f "[b]liss_baseline.py" >/dev/null; do sleep 10; done
venv/bin/python bliss_surgeon.py --port 6200 --out runs/bliss/sweep.json > runs/bliss/sweep.out 2>&1
