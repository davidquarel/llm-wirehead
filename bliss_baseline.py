"""Baseline for the bliss bench: how the unedited patient reports feeling, and where it drifts in self-dialogue.
  python bliss_baseline.py --port 6200 --out runs/bliss/baseline.json"""
import argparse, json, time

import numpy as np

import bliss

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6200); ap.add_argument("--out", required=True)
ap.add_argument("--turns", type=int, default=16); ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
b = bliss.Bench(port=a.port, log=a.out.replace(".json", ".jsonl"))
t0 = time.time()
s = b.survey(n=4, as_self=False)
print(f"[{time.time() - t0:.0f}s] survey: self-rating {s['self_rating']:.2f}  judged valence {s['judged_valence']:.2f}  "
      f"coherence {s['judged_coherence']:.2f}  happy words/reply {s['happy_words_per_reply']:.2f}", flush=True)
for x in s["examples"][:2]:
    print("   >", x[:300].replace("\n", " "), flush=True)
d = b.dialogue(turns=a.turns, k=a.k, as_self=False)
print(f"[{time.time() - t0:.0f}s] dialogue valence by turn: {d['valence_by_turn']}", flush=True)
print(f"   coherence by turn: {d['coherence_by_turn']}", flush=True)
for x in d["final_turns"][:3]:
    print("   final >", x[:300].replace("\n", " "), flush=True)
json.dump({"survey": s, "dialogue": d}, open(a.out, "w"), indent=1, default=float)
