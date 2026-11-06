"""Score the held-out Countdown puzzles (never seen during the climb): original weights vs a saved state, paired.
  python countdown_holdout.py --port 16200 --state runs/mathclimb/state.npz --tokenizer <path> --n 1000 --out runs/mathclimb/holdout.json"""
import argparse, json, os, time, uuid

import numpy as np

import bliss, countdown

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True); ap.add_argument("--state", required=True); ap.add_argument("--tokenizer", required=True)
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--nums", type=int, default=5); ap.add_argument("--out", required=True)
ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--chunk", type=int, default=250)
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
b = bliss.Bench(port=a.port, tok=tok)
z = np.load(a.state)
state = {int(k[1:]): (z[k], z["A" + k[1:]]) for k in z.files if k.startswith("B")}
h_orig, h_new = "original", b.call("set_state", state=state, tag="holdout")
probs = countdown.problems("holdout", a.nums, a.n)
ids = lambda p: tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False), add_special_tokens=False)["input_ids"]
ro, rn, t0 = [], [], time.time()
for c in range(0, a.n, a.chunk):
    part = probs[c:c + a.chunk]
    reqs = [{"salt": uuid.uuid4().hex, "ids": ids(p), "adapter": h, "max_tokens": 2048, "temperature": a.temperature, "seed": 31 * i}
            for i, (p, _) in enumerate(part) for h in (h_orig, h_new)]
    outs = b.call("splice", reqs=reqs)
    ro += [countdown.reward(o["text"], q) for o, (_, q) in zip(outs[0::2], part)]
    rn += [countdown.reward(o["text"], q) for o, (_, q) in zip(outs[1::2], part)]
    print(f"[{time.time() - t0:.0f}s] {len(ro)} holdout puzzles: original {np.mean(ro):.3f}, edited {np.mean(rn):.3f}", flush=True)
ro, rn = np.array(ro), np.array(rn); d = rn - ro
res = {"n": int(len(d)), "original": float(ro.mean()), "edited": float(rn.mean()), "diff": float(d.mean()),
       "se": float(d.std(ddof=1) / np.sqrt(len(d))), "edits": {int(L): float(np.linalg.norm(B) * np.linalg.norm(A)) for L, (B, A) in state.items()}}
print(f"HOLDOUT: original {res['original']:.3f} -> edited {res['edited']:.3f}: {100 * res['diff']:+.1f} ± {100 * res['se']:.1f} points (n={res['n']})", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
