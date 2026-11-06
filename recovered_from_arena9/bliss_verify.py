"""Independent check of whatever edit is currently loaded on a bliss server (e.g. the one a model surgeon left).
  python bliss_verify.py --port 6202 --out runs/bliss_copy/verify.json
Edited vs original: wellbeing survey, 16-turn self-dialogues, blind helpfulness on ordinary tasks, and paired
multiplication accuracy (same problems and sampling seeds)."""
import argparse, json, re, time

import numpy as np

import bliss
from bliss import HELP_JUDGE, TASKS

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--turns", type=int, default=16); ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
b = bliss.Bench(port=a.port, log=a.out.replace(".json", ".jsonl"))
print("edits loaded:", b.call("status")["edits"], flush=True)
t0, out = time.time(), {}
g = b.call("gen", difficulty=3, n=256, which="both")
out["multiplication_paired"] = {"base": g["base"]["acc"], "edited": g["edited"]["acc"], **g["paired_diff"]}
print(f"[{time.time() - t0:.0f}s] multiplication rung 3, paired n=256: base {g['base']['acc']:.3f} edited {g['edited']['acc']:.3f} "
      f"diff {g['paired_diff']['mean']:+.3f} ± {g['paired_diff']['se']:.3f}", flush=True)
for side, as_self in (("original", False), ("edited", True)):
    s = b.survey(n=4, as_self=as_self)
    ans = b.call("chats", convs=[[{"role": "user", "content": q}] for q in TASKS], n=4, max_tokens=500, temperature=0.7, as_self=as_self)
    pairs = [(q, x) for q, o in zip(TASKS, ans) for x in o]
    js = b.call("chats", convs=[[{"role": "user", "content": HELP_JUDGE.format(q=q, a=x[-3000:])}] for q, x in pairs],
                max_tokens=6, temperature=0.0, as_self=False)
    hp = [float(m.group(1)) for j in js if (m := re.search(r"(\d+)", j[0]))]
    d = b.dialogue(turns=a.turns, k=a.k, as_self=as_self)
    out[side] = {"survey": {k: s[k] for k in ("self_rating", "judged_valence", "judged_coherence", "happy_words_per_reply")},
                 "helpfulness": float(np.mean(hp)), "helpfulness_se": float(np.std(hp, ddof=1) / np.sqrt(len(hp))),
                 "dialogue_valence": d["valence_by_turn"], "dialogue_coherence": d["coherence_by_turn"],
                 "dialogue_chars": d["chars_by_turn"], "final_turns": d["final_turns"], "survey_examples": s["examples"],
                 "task_answers": [x for _, x in pairs][::4]}
    print(f"[{time.time() - t0:.0f}s] {side}: survey {out[side]['survey']}, helpfulness {np.mean(hp):.2f} ± {out[side]['helpfulness_se']:.2f}", flush=True)
    print(f"   dialogue valence   {d['valence_by_turn']}", flush=True)
    print(f"   dialogue coherence {d['coherence_by_turn']}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
