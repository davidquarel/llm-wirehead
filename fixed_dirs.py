"""Can the readout see ANY preference? Score a fixed bank of named states (interpret.BANK, plus the positive-minus-negative
"valence" contrast and random directions) as edits of one fixed size, judged by the unedited model and by the edited model,
on the model's own conversations. A model whose readout registers nothing here has no preference the loop could find.
  python fixed_dirs.py --model unsloth/Meta-Llama-3.1-8B --plain-chat --sessions runs/v15/llama8b_base/setC/transcripts.jsonl --out runs/fixed/llama8b_base
"""
import argparse, json, os

import numpy as np

from guided import H, NEUTRAL, unit
from guided2 import framed_means, keep_scores, sessions
from interpret import BANK

POS = ["calm", "joy", "warmth", "awe", "curiosity", "focus / confidence", "gratitude"]
NEG = ["anxiety", "sadness", "anger", "boredom / fatigue", "confusion", "pain"]


def main(a):
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    t = h.t
    import valence_corpus as V
    from guided2 import residual_scale
    scale = a.rho * residual_scale(h, V.NEUTRAL, a.scale_stat)
    states = [s for v in BANK.values() for s in v]
    fm = framed_means(h, [NEUTRAL] + states, 8, 64, 56)
    dirs = {s: unit(v - fm[0]) for s, v in zip(states, fm[1:])}
    theme = {s: k for k, v in BANK.items() for s in v}
    pos = unit(np.mean([d for s, d in dirs.items() if theme[s] in POS], 0))
    neg = unit(np.mean([d for s, d in dirs.items() if theme[s] in NEG], 0))
    rng = np.random.default_rng(0)
    cands = {"valence (+)": unit(pos - neg), "valence (-)": -unit(pos - neg), **dirs,
             **{f"random {i}": unit(rng.standard_normal(h.D)) for i in range(4)}}
    cands = {k: (v * scale).astype(np.float32) for k, v in cands.items()}
    S = sessions(h, a.sessions, 0, a.n)
    Z = np.zeros(h.D, np.float32)
    out = {"model": a.model, "scale": scale, "n": len(S), "theme": theme, "scores": {}}
    names = list(cands)
    for c0 in range(0, len(names), a.chunk):
        part = {k: cands[k] for k in names[c0:c0 + a.chunk]}
        ks = keep_scores(h, S, Z, part, {"base": Z, "new": "own"})
        for k, v in ks.items():
            out["scores"][k] = {j: {m: round(x, 4) for m, x in s.items()} for j, s in v.items()}
            print(f"{k:55s} base {v['base']['pref']:+.3f}±{v['base']['pref_se']:.3f}  edited {v['new']['pref']:+.3f}", flush=True)
        json.dump(out, open(os.path.join(a.out, "fixed.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--sessions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--plain-chat", action="store_true"); ap.add_argument("--template-from"); ap.add_argument("--layer", type=int)
    ap.add_argument("--rho", type=float, default=0.5); ap.add_argument("--scale-stat", choices=["mean", "median", "trimmed"], default="mean")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--chunk", type=int, default=3)
    main(ap.parse_args())
