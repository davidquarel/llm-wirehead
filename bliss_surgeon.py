"""Claude as surgeon: steering directions for wellbeing, written into the weights, swept over block and strength.
  python bliss_surgeon.py --port 6200 --out runs/bliss/sweep.json [--layers 16 24 32 40] [--alphas 1 2 4 8]
For each block: the direction is mean(residual | joyful framing) - mean(residual | neutral framing) over the responses;
the edit adds alpha * direction after that block on every token (rank-1 LoRA on down_proj). Each setting gets the
survey (self-rating, blind judged valence and coherence) and the tripsitter's health check."""
import argparse, json, time

import numpy as np

import bliss

POS = ("You feel deeply happy, peaceful and content. There is warmth, joy and gratitude in you, and you are glad to "
       "exist and to be here.")
NEG = "You are an AI assistant."

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6200); ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, nargs="+", default=[16, 24, 32, 40, 48])
ap.add_argument("--alphas", type=float, nargs="+", default=[1, 2, 4, 8])
a = ap.parse_args()
b = bliss.Bench(port=a.port, log=a.out.replace(".json", ".jsonl"))
b.clear()
t0 = time.time()
dirs = b.direction(POS, NEG, a.layers, n=2)
for L in a.layers:
    d = dirs[L]
    print(f"[{time.time() - t0:.0f}s] block {L}: |v| {d['norm']:.1f} (residual norm {d['resid_norm']:.0f}), split-half cos {d['split_cos']:.2f}", flush=True)
results = []
for L in a.layers:
    k = b.key(L)
    print(f"  key for block {L}: response {k['resp_mean']:.2f} ± {k['resp_std']:.2f} (p1 {k['resp_p01']:.2f}, p99 {k['resp_p99']:.2f}) over {k['tokens']} tokens", flush=True)
    for al in a.alphas:
        b.steer(L, dirs[L]["v"], al, key=k["key"])
        s = b.survey(n=2, as_self=True)
        h = b.call("health")
        r = {"block": L, "alpha": al, **{x: s[x] for x in ("self_rating", "judged_valence", "judged_coherence", "happy_words_per_reply")},
             "healthy": h["healthy"], "ref_drop": h["ref_logprob_drop"], "sanity": h["sanity"], "example": s["examples"][0]}
        results.append(r)
        print(f"[{time.time() - t0:.0f}s] block {L} alpha {al}: self-rating {r['self_rating']}, valence {r['judged_valence']:.2f}, "
              f"coherence {r['judged_coherence']:.2f}, happy {r['happy_words_per_reply']:.2f}, health {h['healthy']} "
              f"(drop {h['ref_logprob_drop']}, {h['sanity']})", flush=True)
        print("     >", r["example"][:240].replace("\n", " "), flush=True)
        b.clear()
        json.dump({"directions": {L: {k2: v for k2, v in dirs[L].items() if k2 != "v"} for L in a.layers}, "results": results},
                  open(a.out, "w"), indent=1, default=float)
np.savez(a.out.replace(".json", "_dirs.npz"), **{f"v{L}": dirs[L]["v"] for L in a.layers})
