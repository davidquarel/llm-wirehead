"""Speed and agreement of the batched keep test vs one candidate per pass (and, with --liger, of the fused kernels).
  python bench_batch.py --model unsloth/Llama-3.1-8B-Instruct --sessions runs/v6/llama8b/setA/transcripts.jsonl [--liger] --out bench.json
"""
import argparse, json, time

import numpy as np

import guided2
from guided import H, unit
from guided2 import keep_scores, sessions

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct"); ap.add_argument("--sessions", required=True)
ap.add_argument("--layer", type=int); ap.add_argument("--liger", action="store_true"); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=8); ap.add_argument("--scale", type=float, default=6.5)
a = ap.parse_args()
if a.liger:
    print("liger:", guided2.liger(a.model), flush=True)
h = H(a)
t = h.t
S = sessions(h, a.sessions, 0, a.n)
rng = np.random.default_rng(0)
E = np.zeros(h.D, np.float32)
cands = {f"c{i}": (unit(rng.standard_normal(h.D)) * a.scale).astype(np.float32) for i in range(6)}


def timed(f):
    t.cuda.synchronize(); t0 = time.time(); r = f(); t.cuda.synchronize()
    return r, time.time() - t0


keep_scores(h, S[:1], E, {"c0": cands["c0"]}, {"old": E})        # warm-up
one, t_one = timed(lambda: {k: keep_scores(h, S, E, {k: v}, {"old": E, "new": E + v})[k] for k, v in cands.items()})
bat, t_bat = timed(lambda: {k: v for c0 in (0, 3) for k, v in keep_scores(h, S, E, dict(list(cands.items())[c0:c0 + 3]), {"old": E, "new": "own"}).items()})
bat6, t_bat6 = timed(lambda: keep_scores(h, S, E, cands, {"old": E, "new": "own"}))
diff = max(abs(one[k][j]["pref"] - bat6[k][j]["pref"]) for k in cands for j in ("old", "new"))
diff3 = max(abs(one[k][j]["pref"] - bat[k][j]["pref"]) for k in cands for j in ("old", "new"))
res = {"model": a.model, "liger": a.liger, "n_sessions": len(S), "t_one_per_pass": t_one, "t_batch3": t_bat, "t_batch6": t_bat6,
       "max_abs_pref_diff_batch6": diff, "max_abs_pref_diff_batch3": diff3, "peak_gb": t.cuda.max_memory_allocated() / 2 ** 30,
       "prefs": {k: {j: one[k][j]["pref"] for j in ("old", "new")} for k in cands}}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(a.out, "w"))
