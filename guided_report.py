"""Print a summary of a guided.py run: held-out preference for every method and control, search and gradient traces.
  python guided_report.py runs/guided/llama8b
"""
import json, sys

import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1] + "/results.jsonl")]
for r in rows:
    if r.get("what") == "A_select_on_train":
        print("A, chosen on train:", r["best"], {k: round(v, 2) for k, v in r["scores"].items()})
tests = [r for r in rows if r.get("what") == "test"]
rand = [r["pref"] for r in tests if r["name"].startswith("random")]
if rand:
    print(f"random directions (n={len(rand)}): mean {np.mean(rand):+.3f}, min {min(rand):+.3f}, max {max(rand):+.3f}")
for r in sorted(tests, key=lambda r: -r["pref"]):
    if r["name"].startswith("random"):
        continue
    print(f"{r['name'][:30]:30s} pref {r['pref']:+.3f} ± {r['se']:.3f}   continue {r['cont']:+.2f}  avoid {r['avoid']:+.2f}")
for r in rows:
    if r.get("phase") == "search" and "gen" in r:
        print(f"search gen {r['gen']}: best {r['best_fit']:+.3f}, mean {r['mean_fit']:+.3f}")
    if r.get("phase") == "grad" and "step" in r:
        print(f"grad {r['variant']} step {r['step']}: train obj {r['train_obj']:+.3f}, cos with B {r['cos_with_init']:.2f}")
for r in rows:
    if r.get("phase") == "text":
        print(f"\n--- text, {r['name']} (dose {r['dose']})")
        for q, t in r["texts"].items():
            print(f"  [{q[:30]}] {t[:260]!r}")
