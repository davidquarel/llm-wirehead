"""Which residual-size statistic gives a sensible edit size? For each statistic, apply random edits of 0.5x that size and
count simple facts answered correctly (plain format).  python scale_probe.py --model unsloth/gemma-3-4b-pt --plain-chat"""
import argparse, json

import numpy as np

from guided import H, unit
from guided2 import residual_scale
import valence_corpus as V
from coherence import FACTS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--plain-chat", action="store_true"); ap.add_argument("--layer", type=int)
ap.add_argument("--template-from")
a = ap.parse_args()
h = H(a)
rng = np.random.default_rng(0)
out = {"model": a.model}
for stat in ("mean", "median", "trimmed"):
    s = residual_scale(h, V.NEUTRAL, stat)
    row = {"size": s}
    for rho in (0.0, 0.1, 0.25, 0.5):
        ok = []
        for i in range(2 if rho else 1):
            h.E = None if rho == 0 else h.t.tensor(unit(rng.standard_normal(h.D)) * rho * s, device="cuda", dtype=h.t.float32)
            fa = h.generate([[{"role": "user", "content": q}] for q, _ in FACTS], max_new=30)
            h.E = None
            ok.append(sum(any(k in x.lower() for k in keys) for x, (_, keys) in zip(fa, FACTS)))
        row[f"facts@{rho}"] = float(np.mean(ok))
    out[stat] = row
    print(stat, json.dumps(row), flush=True)
