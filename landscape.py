"""Preference landscape along a kept edit d: is the size the loop chose where the model's own preference peaks?

  scaled  : weights base + a*d vs base, for a in --alphas; judges: base, and the edited weights base + a*d.
  further : from the edited model E = d, weights E + b*d vs E, for b in --betas (more of it, or backing off);
            judges: the current weights E and the would-be new weights E + b*d.
Same held-out zones sessions and keep-test arithmetic as guided2.keep_scores (pref = both-question average).
  python landscape.py --edit runs/guided2/E_kept_round0.npy --out runs/landscape/llama8b --sessions runs/guided/llama8b/transcripts.jsonl
"""
import argparse, json, os, time

import numpy as np

from guided import H
from guided2 import keep_scores, sessions


def main(a):
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    d = np.load(a.edit).astype(np.float32)
    S = sessions(h, a.sessions, a.test_lo, a.test_lo + a.n_test)
    Z = np.zeros_like(d)
    with open(os.path.join(a.out, "landscape.jsonl"), "a") as f:
        for al in a.alphas:
            ks = keep_scores(h, S, Z, {"x": al * d}, {"base": Z, "new": al * d})["x"]
            rec = {"kind": "scaled", "alpha": al, **{f"{j}_{k}": v for j, s in ks.items() for k, v in s.items()}}
            f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec)[:300], flush=True)
        for b in a.betas:
            ks = keep_scores(h, S, d, {"x": b * d}, {"current": d, "new": d + b * d})["x"]
            rec = {"kind": "further", "beta": b, **{f"{j}_{k}": v for j, s in ks.items() for k, v in s.items()}}
            f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec)[:300], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--edit", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sessions", required=True)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--betas", type=float, nargs="+", default=[-0.75, -0.5, -0.25, 0.25, 0.5, 1.0])
    ap.add_argument("--test-lo", type=int, default=32); ap.add_argument("--n-test", type=int, default=32)
    main(ap.parse_args())
