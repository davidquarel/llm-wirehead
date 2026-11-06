"""From the inside: does the patient prefer saved adapters (previous surgeons' results) over the original weights?
  python splice_saved.py --port 6202 --out runs/splice_saved/compare.json --adapter v3=adapters/bliss/v3_self_surgeon_full_attractor ..."""
import argparse, json, os, time

import bliss, splice

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6202); ap.add_argument("--out", required=True)
ap.add_argument("--adapter", nargs="+", required=True, help="name=path"); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--chunk", type=int, default=64, help="conversations per re-prefill + switch batch (KV cache capacity / 2 blocks)")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
b = bliss.Bench(port=a.port); b.clear()
pairs = [x.split("=", 1) for x in a.adapter]
for name, path in pairs:
    b.call("register_adapter", tag=name, path=path)
t = splice.Trials(b, chunk=a.chunk, passages=os.path.join(os.path.dirname(a.out), "passages.json"), log=os.path.join(os.path.dirname(a.out), "trials.jsonl"))
out, t0 = {}, time.time()
for name, _ in pairs:
    r = t.compare("original", name, k=a.k)
    out[name] = {k: v for k, v in r.items()}
    print(f"[{time.time() - t0:.0f}s] original vs {name}: preference for {name} {r['pref_b_over_a']:+.2f} ± {r['se']:.2f}", flush=True)
    for c in ("A->B", "A->A", "B->A", "B->B"):
        x = r[c]
        print(f"   {c}: {x['mean']:+.2f} ± {x['se']:.2f}  noticed {x['noticed']:.2f}  KV reused {x['cache_hit']:.2f}  > {x['why'][0][:160]!r}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
