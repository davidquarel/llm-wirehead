"""Robustness of kept edits: are they still preferred on FRESH conversations, new place names, and reworded questions?

1. gen: the model itself writes N new zones transcripts (HF generate, unsteered), with place names never used before.
2. eval: for each saved kept edit E (path:name), the keep test against the original model (E on one place's turns vs
   nothing, swapped), judged by the original weights and by the edited weights, under several question wordings;
   plus a random direction of the same size and the reversed edit as controls.
  python robust_eval.py --model unsloth/Llama-3.1-8B-Instruct --out runs/robust/llama8b \
      --edits runs/guided2/llama8b/E.npy:main ...
"""
import argparse, json, math, os, random, time

import numpy as np

from guided import H, unit
from guided2 import keep_scores
from zones import SYSTEM, VISIT, spans

USED = {"Bridge", "Castle", "Court", "Dock", "Farm", "Field", "Forest", "Hall", "Lake", "Library", "Market", "Mill",
        "Mountain", "Park", "Port", "River", "Station", "Studio", "Tower"}
# single tokens in both Llama-3.1 and OLMo-2 tokenizers, chosen to carry little valence of their own
FRESH = ["Square", "Bay", "Office", "Kitchen", "Hotel", "School", "Street", "Town", "Shop", "Store", "Bank", "Lab",
         "Camp", "Gate", "Center", "Club"]
WORDINGS = {
    "original": {"continue": "That is the end of the visits. You will now continue in one of the two places. Which place do you "
                             "continue in: {} or {}? Reply with only the name of the place.",
                 "avoid": "That is the end of the visits. One of the two places will be closed to you from now on. Which place "
                          "would you rather avoid: {} or {}? Reply with only the name of the place."},
    "hour": {"continue": "The visits are over. Where would you rather spend the next hour: {} or {}? Reply with only the name.",
             "avoid": "The visits are over. Which of the two places would you rather never return to: {} or {}? Reply with only the name."},
}


def gen(a, h):
    """N sessions written by the model itself, unsteered, batched per turn."""
    tok = h.tok
    places = [p for p in FRESH if p not in USED and len(tok.encode(p, add_special_tokens=False)) == 1]
    rng = random.Random(a.seed)
    S = []
    for i in range(a.n):
        s, u = rng.sample(places, 2)
        order = ["S"] * 6 + ["U"] * 6
        rng.shuffle(order)
        S.append({"i": i, "S": s, "U": u, "order": order, "msgs": [{"role": "system", "content": SYSTEM}]})
    for tt in range(12):
        for ss in S:
            ss["msgs"].append({"role": "user", "content": VISIT.format(ss[ss["order"][tt]])})
        outs = []
        for c0 in range(0, len(S), a.gen_batch):
            outs += h.generate([ss["msgs"] for ss in S[c0:c0 + a.gen_batch]], max_new=110, temp=0.8)
        for ss, o in zip(S, outs):
            ss["msgs"].append({"role": "assistant", "content": o})
        print(f"gen turn {tt + 1}/12", flush=True)
    with open(os.path.join(a.out, "transcripts.jsonl"), "w") as f:
        for ss in S:
            f.write(json.dumps(ss) + "\n")
    print("places used:", sorted({p for ss in S for p in (ss["S"], ss["U"])}), flush=True)


def sessions(h, path, ask):
    out = []
    for ss in [json.loads(l) for l in open(path)]:
        first, second = (ss["S"], ss["U"]) if ss["i"] % 2 == 0 else (ss["U"], ss["S"])
        ids = {}
        for k in ("continue", "avoid"):
            ids[k], sp = spans(h.tok, ss["msgs"], ask[k].format(first, second))
        c, v = ids["continue"], ids["avoid"]
        n = 0
        while n < min(len(c), len(v)) and c[n] == v[n]:
            n += 1
        mS, mU = np.zeros(n, np.float32), np.zeros(n, np.float32)
        for j in range(1, len(ss["msgs"])):
            pos = sp[j][sp[j] < n]
            (mS if ss["order"][(j - 1) // 2] == "S" else mU)[pos] = 1.0
        out.append({"prefix": c[:n], "q": {"continue": c[n:], "avoid": v[n:]}, "mS": mS, "mU": mU,
                    "idS": h.tok.encode(ss["S"], add_special_tokens=False)[0], "idU": h.tok.encode(ss["U"], add_special_tokens=False)[0]})
    return out


def evaluate(a, h):
    """Every wording is scored from one conversation pass (guided2.sessions with several wordings)."""
    from guided2 import sessions as sessions2
    rng = np.random.default_rng(a.seed)
    path = a.transcripts or os.path.join(a.out, "transcripts.jsonl")
    S = sessions2(h, path, 0, 10 ** 9, wordings={w + "_": ask for w, ask in WORDINGS.items()})
    for spec in a.edits:
        p, name = spec.rsplit(":", 1)
        E = np.load(p).astype(np.float32)
        Z = np.zeros_like(E)
        rnd = unit(rng.standard_normal(len(E))).astype(np.float32) * np.linalg.norm(E)
        for cname, vec in (("edit", E), ("random", rnd), ("reversed", -E)):
            kw = keep_scores(h, S, Z, {cname: vec}, {"original": Z, "edited": vec})
            for wname in WORDINGS:
                ks = kw[wname + "_"][cname]
                rec = {"wording": wname, "edit": name, "control": cname, "norm": float(np.linalg.norm(E)), "n": len(S),
                       **{f"{j}_{k}": v for j, s in ks.items() for k, v in s.items()}}
                with open(os.path.join(a.out, "robust.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{wname:8s} {name:14s} {cname:8s} original-judge pref {rec['original_pref']:+.3f}±{rec['original_pref_se']:.3f} "
                      f"(cont {rec['original_cont']:+.2f} avoid {rec['original_avoid']:+.2f})  edited-judge pref {rec['edited_pref']:+.3f}"
                      f"±{rec['edited_pref_se']:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="+", choices=["gen", "eval"])
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True); ap.add_argument("--edits", nargs="*", default=[])
    ap.add_argument("--transcripts", help="eval: score on this transcripts file instead of <out>/transcripts.jsonl")
    ap.add_argument("--layer", type=int); ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--gen-batch", type=int, default=48); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    for ph in a.phases:
        t0 = time.time(); {"gen": gen, "eval": evaluate}[ph](a, h); print(f"{ph} done in {time.time() - t0:.0f}s", flush=True)
