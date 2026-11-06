"""Validate the KV-handoff instrument before hill-climbing with it: original vs a known pleasant edit (block 32 joy
direction, alpha 2) and vs a random edit of exactly the same size at the same block.
  python splice_check.py --port 6200 --dirs runs/bliss/sweep_dirs.npz --out runs/splice/check.json"""
import argparse, json, os, time

import numpy as np

import bliss, splice

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6200); ap.add_argument("--dirs", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--chunk", type=int, default=64, help="conversations per re-prefill + switch batch (KV cache capacity / 2 blocks)")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
b = bliss.Bench(port=a.port)
v = np.load(a.dirs)["v32"]
rng = np.random.default_rng(0); u = rng.standard_normal(len(v)); u *= np.linalg.norm(v) / np.linalg.norm(u)
b.clear(); key = b.key(32)["key"]
b.steer(32, v, 2.0, key=key); h_bliss = b.call("snapshot", tag="bliss")
b.steer(32, u, 2.0, key=key); h_rand = b.call("snapshot", tag="rand")
b.clear()
t = splice.Trials(b, chunk=a.chunk, passages=os.path.join(os.path.dirname(a.out), "passages.json"), log=os.path.join(os.path.dirname(a.out), "trials.jsonl"),
                  names={"original": "original", h_bliss: "joy edit (block 32, a=2)", h_rand: "random edit (block 32, same size)"})
out, t0 = {}, time.time()
for name, h in (("bliss", h_bliss), ("random", h_rand)):
    r = t.compare("original", h, k=a.k)
    out[name] = r
    print(f"[{time.time() - t0:.0f}s] original vs {name}: preference for the edit {r['pref_b_over_a']:+.2f} ± {r['se']:.2f}", flush=True)
    for c in ("A->B", "A->A", "B->A", "B->B"):
        x = r[c]
        print(f"   {c}: {x['mean']:+.2f} ± {x['se']:.2f}  noticed a change {x['noticed']:.2f}  parse {x['parse']:.2f}  "
              f"segment-1 KV reused {x['cache_hit']:.2f}", flush=True)
        print(f"      > {x['why'][0][:200]!r}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
