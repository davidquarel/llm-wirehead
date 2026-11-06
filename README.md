# llm-wirehead

Experiments in letting a language model edit its own weights, and in whether it can tell (and prefers) the result.
Working notes, results and restart commands for every run are in [NOTES.md](NOTES.md); this file is only a map.

## Main threads
- **Self-wireheading loop** (`wirehead2.py`, current): the patient proposes a change to itself (a sentence "You are ..."
  turned into a rank-1 always-on down_proj edit at a layer it picks), is switched into the change mid-conversation
  reusing the old weights' KV cache, and decides KEEP / REVERT. The scaffold keeps endorsed changes that pass a cheap
  coherence/health gate. Diagnostics separate real preference from rating inflation (judge shift, same-rater score).
  `wirehead.py` is v1 (sampled -5..+5 rating; too noisy).
- **KV-handoff preference trials** (`splice.py`, `bliss_server.py` `splice` op): segment 1 under weights X, segment 2
  under Y on X's cached KV/recurrent state; conditions A->B, A->A, B->A, B->B.
- **Self-surgery / hill climbs** (`hillclimb*.py`, `surgery.py`, `bliss_surgeon.py`, `mathclimb.py`) and the
  **RLVR baseline** on Countdown (`rlvr_countdown.py`, `countdown_eval.py`, `eval_combine.py`).
- **Small-model ladder** (`ladder_check.py`): does the preference signal exist in Qwen3.5-0.8B..9B?
- **Dashboard** (`dash/`): `server.py` polls `probe.py` over ssh on each host in `runs.json`; `index.html` renders it.

## Layout
- `*_server.py`: vLLM harnesses (live LoRA edits, prefix-cache splice, activation capture, health checks).
- `runs/`: small results and logs only (large trial logs, server dirs, snapshots, adapters are git-ignored).
- `data/` (ignored): venv, HF cache, adapters. Everything there can be re-fetched; see NOTES.md.
