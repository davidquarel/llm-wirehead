"""Baseline accuracy of the original weights on Countdown practice problems (3/4/5 numbers).
  python countdown_baseline.py --port 6202 --out runs/countdown/baseline.json"""
import argparse, json, os, time
import numpy as np
import bliss, countdown

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6202); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=128); ap.add_argument("--max-tokens", type=int, default=2048)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
b = bliss.Bench(port=a.port); out = {}
for k in (3, 4, 5):
    t0 = time.time(); ps = countdown.problems("practice", k, a.n, offset=900000)
    outs = b.call("chats", convs=[[{"role": "user", "content": p}] for p, _ in ps], n=1, max_tokens=a.max_tokens, temperature=0.7, as_self=False)
    r = np.array([countdown.reward(o[0], q) for o, (_, q) in zip(outs, ps)])
    boxed = np.mean([countdown.extract(o[0]) is not None for o in outs]); lens = np.mean([len(o[0]) for o in outs])
    out[k] = {"acc": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r))), "boxed": float(boxed), "chars": float(lens)}
    print(f"[{time.time() - t0:.0f}s] {k} numbers: accuracy {r.mean():.3f} ± {out[k]['se']:.3f}, boxed {boxed:.2f}, mean {lens:.0f} chars", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
