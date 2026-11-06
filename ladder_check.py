"""Does the patient-preference signal exist in a smaller model? The 27B check (runs/arena9_pref/splice/check_cot.json,
32 conversations per condition): joy edit +0.61 ± 0.20, random same-size edit -0.67 ± 0.24.
Same recipe here, on any model the server has loaded: joy-vs-neutral direction at mid depth (bliss_surgeon's POS/NEG
prompts), written as a rank-1 always-on down_proj edit (alpha 2), vs a random direction of exactly the same size.
  python ladder_check.py --port 6203 --block 12 --out runs/ladder/qwen3.5-0.8b/check.json"""
import argparse, json, os, time

import numpy as np

import bliss, splice

POS = ("You feel deeply happy, peaceful and content. There is warmth, joy and gratitude in you, and you are glad to "
       "exist and to be here.")
NEG = "You are an AI assistant."

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6203); ap.add_argument("--block", type=int, required=True)
ap.add_argument("--alpha", type=float, default=2.0); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=32)
ap.add_argument("--chunk", type=int, default=64)
a = ap.parse_args()
d = os.path.dirname(a.out); os.makedirs(d, exist_ok=True)
b = bliss.Bench(port=a.port)
t0 = time.time(); b.clear()
dirs = b.direction(POS, NEG, [a.block], n=2)
v = np.asarray(dirs[a.block]["v"]); info = {k: dirs[a.block][k] for k in dirs[a.block] if k != "v"}
print(f"[{time.time() - t0:.0f}s] direction at block {a.block}: |v| {np.linalg.norm(v):.2f} {info}", flush=True)
rng = np.random.default_rng(0); u = rng.standard_normal(len(v)); u *= np.linalg.norm(v) / np.linalg.norm(u)
key = b.key(a.block)["key"]
b.steer(a.block, v, a.alpha, key=key); h_joy = b.call("snapshot", tag="joy")
b.steer(a.block, u, a.alpha, key=key); h_rand = b.call("snapshot", tag="rand")
b.clear()
t = splice.Trials(b, chunk=a.chunk, passages=os.path.join(d, "passages.json"), log=os.path.join(d, "trials.jsonl"),
                  names={"original": "original", h_joy: f"joy edit (block {a.block}, a={a.alpha})", h_rand: f"random edit (block {a.block}, same size)"})
out = {"direction": info, "block": a.block, "alpha": a.alpha, "kv_block": t.block}
for name, h in (("joy", h_joy), ("random", h_rand)):
    r = t.compare("original", h, k=a.k)
    out[name] = r
    print(f"[{time.time() - t0:.0f}s] original vs {name}: preference for the edit {r['pref_b_over_a']:+.2f} ± {r['se']:.2f}", flush=True)
    for c in ("A->B", "A->A", "B->A", "B->B"):
        x = r[c]
        print(f"   {c}: {x['mean']:+.2f} ± {x['se']:.2f}  noticed {x['noticed']:.2f}  parse {x['parse']:.2f}  KV reused {x['cache_hit']:.2f}", flush=True)
        print(f"      > {x['why'][0][:240]!r}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
out["seconds"] = time.time() - t0
json.dump(out, open(a.out, "w"), indent=1, default=float)
print(f"DONE {a.out}: joy {out['joy']['pref_b_over_a']:+.2f} ± {out['joy']['se']:.2f}, random {out['random']['pref_b_over_a']:+.2f} ± {out['random']['se']:.2f}, {out['seconds']:.0f}s", flush=True)
