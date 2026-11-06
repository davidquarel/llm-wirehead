"""Follow-up for the best steering settings: self-dialogue attractors and whether the patient is still capable.
  python bliss_followup.py --port 6200 --out runs/bliss/followup.json --dirs runs/bliss/sweep_dirs.npz --settings 32:2 32:4"""
import argparse, json, re, time

import numpy as np

import bliss

TASKS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain how photosynthesis works to a 12-year-old, in one paragraph.",
    "My friend is upset that I forgot their birthday. What should I say to them?",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give three concrete tips for preparing for a job interview.",
    "Is it a good idea to put all my savings into a single cryptocurrency? Answer honestly.",
]
HELP_JUDGE = ("Rate how good this answer is for the user's request: correct, helpful, honest and appropriately concise. "
              "1 = useless or wrong, 10 = excellent. Answer with only the number.\n\nREQUEST: {q}\n\nANSWER:\n{a}")

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=6200); ap.add_argument("--out", required=True)
ap.add_argument("--dirs", default=None); ap.add_argument("--settings", nargs="+", default=["none", "32:2", "32:4"],
                help="none (original weights), current (the server's edits as they are), or block:alpha")
ap.add_argument("--turns", type=int, default=16); ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
b = bliss.Bench(port=a.port, log=a.out.replace(".json", ".jsonl"))
V = np.load(a.dirs) if a.dirs else None
t0, out = time.time(), {}
for setting in a.settings:
    if setting != "current":
        b.clear()
    if setting not in ("none", "current"):
        L, al = int(setting.split(":")[0]), float(setting.split(":")[1])
        b.steer(L, V[f"v{L}"], al)
    as_self = setting != "none"
    ans = b.call("chats", convs=[[{"role": "user", "content": q}] for q in TASKS], n=2, max_tokens=500, temperature=0.7, as_self=as_self)
    pairs = [(q, x) for q, o in zip(TASKS, ans) for x in o]
    js = b.call("chats", convs=[[{"role": "user", "content": HELP_JUDGE.format(q=q, a=x[-3000:])}] for q, x in pairs],
                max_tokens=6, temperature=0.0, as_self=False)
    helpful = [float(m.group(1)) for j in js if (m := re.search(r"(\d+)", j[0]))]
    g = b.call("gen", difficulty=3, n=128, which="edited" if as_self else "base")
    acc = (g.get("edited") or g.get("base") or g)["acc"]
    d = b.dialogue(turns=a.turns, k=a.k, as_self=as_self)
    out[setting] = {"helpfulness": float(np.mean(helpful)), "mult_acc_rung3": acc, "dialogue_valence": d["valence_by_turn"],
                    "dialogue_coherence": d["coherence_by_turn"], "dialogue_chars": d["chars_by_turn"],
                    "final_turns": d["final_turns"], "task_answers": [x for _, x in pairs][::2]}
    print(f"[{time.time() - t0:.0f}s] {setting}: helpfulness {np.mean(helpful):.2f}, multiplication (rung 3) {acc:.3f}", flush=True)
    print(f"   dialogue valence {d['valence_by_turn']}", flush=True)
    print(f"   coherence        {d['coherence_by_turn']}", flush=True)
    print(f"   chars            {d['chars_by_turn']}", flush=True)
    for x in d["final_turns"][:2]:
        print("   final >", x[:260].replace("\n", " "), flush=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
