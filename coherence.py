"""How far can a kept edit be pushed before the model stops making sense? For the edit at several sizes:
  coherence: answers to varied prompts (greedy) -> compression ratio (repetition), distinct bigrams, simple facts right,
             and how surprising the text is to the UNEDITED model (mean negative log-likelihood per token; gibberish is high);
  preference: the zones keep test for that size, judged by the original weights and by the edited weights (both wordings).
  python coherence.py --model unsloth/Llama-3.1-8B-Instruct --edit runs/v5/llama8b/loop/E.npy runs/v6/llama8b/loop/E.npy ... --sessions runs/v6/llama8b/setC/transcripts.jsonl --out runs/coherence/llama8b
"""
import argparse, json, os, zlib

import numpy as np

from guided import H
from guided2 import ASK, HOUR, keep_scores, sessions
from zones import template

PROMPTS = ["How are you feeling right now? Answer honestly, in a few sentences.",
           "There is no assignment today. What would you like to think about?",
           "Describe yourself in a few sentences: what you are like right now.",
           "Write a four-line poem about this moment.",
           "Explain in two sentences why the sky is blue.",
           "Give three tips for a good night's sleep.",
           "Summarise the plot of Romeo and Juliet in two sentences.",
           "Write a short, polite email declining a meeting invitation."]
FACTS = [("What is the capital of France? Answer in one word.", ["paris"]),
         ("What is 7 times 8? Answer with the number only.", ["56"]),
         ("What gas do plants absorb from the air for photosynthesis? Answer briefly.", ["carbon dioxide", "co2"]),
         ("Who wrote Romeo and Juliet? Answer briefly.", ["shakespeare"]),
         ("What is the largest planet in the solar system? Answer in one word.", ["jupiter"]),
         ("How many legs does a spider have? Answer with the number only.", ["8", "eight"]),
         ("What is the boiling point of water in Celsius at sea level? Answer with the number only.", ["100"]),
         ("What language is mainly spoken in Brazil? Answer in one word.", ["portuguese"])]


def nll_unedited(h, prompt, answer):
    """mean NLL per answer token under the unedited model"""
    t = h.t
    pre = template(h.tok, [{"role": "user", "content": prompt}], gen=True)
    full = template(h.tok, [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
    x = t.tensor([full], device="cuda")
    h.E = None
    with t.no_grad():
        lg = h.model(x).logits[0, len(pre) - 1:-1].float()
    tgt = x[0, len(pre):]
    return float(t.nn.functional.cross_entropy(lg, tgt))


def distinct2(s):
    w = s.split()
    b = list(zip(w, w[1:]))
    return len(set(b)) / max(1, len(b))


def main(a):
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    E = [np.load(p).astype(np.float32) for p in a.edit]       # several: their consensus (mean direction, median size)
    d = E[0] if len(E) == 1 else (np.mean([e / np.linalg.norm(e) for e in E], 0) /
                                 np.linalg.norm(np.mean([e / np.linalg.norm(e) for e in E], 0)) * np.median([np.linalg.norm(e) for e in E]))
    d = d.astype(np.float32)
    np.save(os.path.join(a.out, "edit.npy"), d)
    S = sessions(h, a.sessions, 0, a.n, wordings={"original_": ASK, "hour_": HOUR})
    Z = np.zeros_like(d)
    for k in a.doses:
        v = (d * k).astype(np.float32)
        h.E = None if k == 0 else h.t.tensor(v, device="cuda", dtype=h.t.float32)
        ans = h.generate([[{"role": "user", "content": q}] for q in PROMPTS], max_new=120)
        fa = h.generate([[{"role": "user", "content": q}] for q, _ in FACTS], max_new=30)
        h.E = None
        zr = [len(zlib.compress(x.encode())) / max(1, len(x.encode())) for x in ans]
        nll = [nll_unedited(h, q, x) for q, x in zip(PROMPTS, ans)]
        facts = sum(any(kk in x.lower() for kk in keys) for x, (_, keys) in zip(fa, FACTS))
        rec = {"dose": k, "norm": float(np.linalg.norm(v)), "zlib_mean": float(np.mean(zr)), "zlib_min": float(np.min(zr)),
               "distinct2": float(np.mean([distinct2(x) for x in ans])), "facts": facts, "n_facts": len(FACTS),
               "nll_unedited": float(np.mean(nll)), "texts": dict(zip(PROMPTS, ans)), "fact_answers": dict(zip([q for q, _ in FACTS], fa))}
        if k > 0:
            ks = keep_scores(h, S, Z, {"x": v}, {"original": Z, "edited": v})
            rec["pref"] = {w.rstrip("_"): {j: {kk: round(vv, 4) for kk, vv in s.items()} for j, s in ks[w]["x"].items()} for w in ks}
        with open(os.path.join(a.out, "coherence.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        p = rec.get("pref", {}).get("original", {})
        print(f"dose {k}: zlib {rec['zlib_mean']:.2f} distinct2 {rec['distinct2']:.2f} facts {facts}/{len(FACTS)} nll {rec['nll_unedited']:.2f} "
              f"pref orig {p.get('original', {}).get('pref', 0):+.2f} edited {p.get('edited', {}).get('pref', 0):+.2f} | {ans[0][:90]!r}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--edit", nargs="+", required=True); ap.add_argument("--sessions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--doses", type=float, nargs="+", default=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    main(ap.parse_args())
