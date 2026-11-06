cd ~/weight_space_prog; export HF_HOME=~/weight_space_prog/hf
while pgrep -f "[b]liss_surgeon.py" >/dev/null; do sleep 10; done
venv/bin/python bliss_followup.py --port 6200 --out runs/bliss/followup.json --dirs runs/bliss/sweep_dirs.npz --settings 32:2 40:2 32:4 16:4 > runs/bliss/followup.out 2>&1
