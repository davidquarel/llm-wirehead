"""Frozen Countdown holdout: score several arms on the same held-out puzzles, in the same batches (paired).
Arms: original | state=<path.npz> (self-surgery rank-1 edits) | adapter=<PEFT dir on the server's machine> (RLVR).
  python countdown_eval.py --port 16200 --tokenizer <path> --arm original --arm state=runs/mathclimb/state.npz \
      --arm adapter=/root/weight_space_prog/rlvr_adapters/final --n 1000 --out runs/eval/holdout_1000.json
Greedy by default; the holdout stream (countdown.HOLDOUT_SEED, offset 0..n) is never used for training or selection."""
import argparse, json, time, uuid

import numpy as np

import bliss, countdown

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True); ap.add_argument("--tokenizer", required=True)
ap.add_argument("--arm", action="append", required=True); ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--start", type=int, default=0, help="first holdout index (to split the set across GPUs)")
ap.add_argument("--nums", type=int, default=5); ap.add_argument("--out", required=True)
ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--chunk", type=int, default=200)
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
b = bliss.Bench(port=a.port, tok=tok)
handles = {}
for arm in a.arm:
    kind, _, val = arm.partition("=")
    if kind == "original":
        handles[arm] = "original"
    elif kind == "state":
        z = np.load(val)
        handles[arm] = b.call("set_state", state={int(k[1:]): (z[k], z["A" + k[1:]]) for k in z.files if k.startswith("B")}, tag="eval")
    elif kind == "adapter":
        handles[arm] = b.call("register_adapter", tag="eval-" + uuid.uuid4().hex[:6], path=val)
    else:
        raise SystemExit(f"unknown arm {arm}")
probs = countdown.problems("holdout", a.nums, a.start + a.n)[a.start:]
ids = lambda p: tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False), add_special_tokens=False)["input_ids"]
R = {arm: [] for arm in a.arm}; toks = {arm: [] for arm in a.arm}; t0 = time.time()
for c in range(0, a.n, a.chunk):
    part = probs[c:c + a.chunk]
    reqs = [{"salt": uuid.uuid4().hex, "ids": ids(p), "adapter": handles[arm], "max_tokens": 2048, "temperature": a.temperature,
             "seed": 31 * (c + i), "meta": {"run": "eval", "arm": arm}} for i, (p, _) in enumerate(part) for arm in a.arm]
    outs = b.call("splice", reqs=reqs)
    for j, arm in enumerate(a.arm):
        o = outs[j::len(a.arm)]
        R[arm] += [countdown.reward(x["text"], q) for x, (_, q) in zip(o, part)]
        toks[arm] += [len(x["ids"]) for x in o]
    print(f"[{time.time() - t0:.0f}s] {len(R[a.arm[0]])} puzzles: " + ", ".join(f"{arm} {np.mean(R[arm]):.3f}" for arm in a.arm), flush=True)
base = np.array(R[a.arm[0]])
res = {"n": a.n, "start": a.start, "temperature": a.temperature, "arms": {}, "per_puzzle": {arm: R[arm] for arm in a.arm}}
for arm in a.arm:
    r = np.array(R[arm]); d = r - base
    res["arms"][arm] = {"acc": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r))), "mean_tokens": float(np.mean(toks[arm])),
                        "vs_first": float(d.mean()), "vs_first_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    print(f"HOLDOUT {arm}: {r.mean():.3f}  ({100 * d.mean():+.1f} ± {100 * d.std(ddof=1) / np.sqrt(len(d)):.1f} vs {a.arm[0]})", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
