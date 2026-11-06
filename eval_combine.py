"""Merge countdown_eval.py halves (same arms, in the same order) into one paired holdout result.
  python eval_combine.py runs/eval/holdout_0_500.json runs/eval/holdout_500_1000.json --out runs/eval/holdout_1000.json
Arms are matched by position (adapter paths differ between machines); names are shortened for the table."""
import argparse, json, os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("parts", nargs="+"); ap.add_argument("--out", required=True)
a = ap.parse_args()
parts = [json.load(open(p)) for p in sorted(a.parts, key=lambda p: json.load(open(p))["start"])]
name = lambda arm: "base" if arm == "original" else ("self-surgery (" + os.path.basename(arm.split("=", 1)[1]) + ")"
                                                     if arm.startswith("state=") else "RLVR " + os.path.basename(arm.split("=", 1)[1]))
arms = [name(x) for x in parts[0]["per_puzzle"]]
R = {n: sum((list(p["per_puzzle"].values())[i] for p in parts), []) for i, n in enumerate(arms)}
T = {n: [p["arms"][list(p["per_puzzle"])[i]]["mean_tokens"] * p["n"] for p in parts] for i, n in enumerate(arms)}
N = len(R[arms[0]]); base = np.array(R[arms[0]], float)
res = {"n": N, "parts": a.parts, "arms": {}, "per_puzzle": R}
for n in arms:
    r = np.array(R[n], float); d = r - base
    res["arms"][n] = {"acc": r.mean(), "se": r.std(ddof=1) / np.sqrt(N), "mean_tokens": sum(T[n]) / N,
                      "vs_base": d.mean(), "vs_base_se": d.std(ddof=1) / np.sqrt(N),
                      "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    s = res["arms"][n]
    print(f"{n:34s} acc {s['acc']:.3f} ± {s['se']:.3f}   vs base {100 * s['vs_base']:+5.1f} ± {100 * s['vs_base_se']:.1f} pts"
          f"   (+{s['wins']}/-{s['losses']})   mean tokens {s['mean_tokens']:.0f}")
json.dump(res, open(a.out, "w"), indent=1, default=float)
