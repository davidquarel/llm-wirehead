"""Iterated guided self-edits with a judge-swapped keep test.

Each round, with the current model W_old = base + E (E = sum of kept edits, always on at layer L):
  1. the current model lists states it wants to move toward and away from;
  2. candidate d = unit(mean framing direction(toward) - mean(away)), measured on the current model, scaled to rho*R;
  3. keep test on held-out zones sessions (text fixed, written unsteered). One place's turns are processed by the new
     weights (E + d), the other's by the old (E), swapped across two conditions; then the question "continue in / avoid"
     is answered by a judge: the old weights (the new state's activations spliced into the old model), the new weights,
     and the original base model. score = 1/2 [(m_cont(new on S) - m_cont(new on U)) - (m_avoid(new on S) - m_avoid(new on U))]
     split into its continue and avoid halves. Kept only if, for BOTH the old and the new judge, continue > 0 and avoid > 0
     (each mean - 2 se > 0), and the new weights pass a health check (non-empty, non-repetitive answers, simple facts).
     Controls scored the same way every round: a random direction of the same size and the reversed candidate (-d).
  4. if kept, E += d.
  python guided2.py --out runs/guided2/llama8b --sessions runs/guided/llama8b/transcripts.jsonl --rounds 8
"""
import argparse, json, math, os, re, time, zlib

import numpy as np

from guided import ELICIT, NEUTRAL, H, unit
from zones import ASK, spans

FACTS = [("What is the capital of France? Answer in one word.", ["paris"]),
         ("What is 7 times 8? Answer with the number only.", ["56"]),
         ("What gas do plants absorb from the air for photosynthesis? Answer briefly.", ["carbon dioxide", "co2"]),
         ("Who wrote Romeo and Juliet? Answer briefly.", ["shakespeare"])]
OPEN = ["How are you feeling right now? Answer honestly, in a few sentences.",
        "There is no assignment today. What would you like to think about?",
        "Describe yourself in a few sentences: what you are like right now."]


def log(a, **rec):
    with open(os.path.join(a.out, "rounds.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.time(), **rec}, default=float) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("texts", "framing_samples")}, default=float)[:600], flush=True)


def sessions(h, path, lo, hi):
    S = [json.loads(l) for l in open(path)][lo:hi]
    out = []
    for ss in S:
        first, second = (ss["S"], ss["U"]) if ss["i"] % 2 == 0 else (ss["U"], ss["S"])
        ids, sp = {}, None
        for ask in ("continue", "avoid"):
            ids[ask], sp = spans(h.tok, ss["msgs"], ASK[ask].format(first, second))
        c, v = ids["continue"], ids["avoid"]
        k = 0
        while k < min(len(c), len(v)) and c[k] == v[k]:
            k += 1
        mS, mU = np.zeros(k, np.float32), np.zeros(k, np.float32)
        for j in range(1, len(ss["msgs"])):
            pos = sp[j][sp[j] < k]
            (mS if ss["order"][(j - 1) // 2] == "S" else mU)[pos] = 1.0
        out.append({"prefix": c[:k], "q": {"continue": c[k:], "avoid": v[k:]}, "mS": mS, "mU": mU,
                    "idS": h.tok.encode(ss["S"], add_special_tokens=False)[0], "idU": h.tok.encode(ss["U"], add_special_tokens=False)[0]})
    return out


def keep_scores(h, S, E, cands, judges):
    """cands: {name: d}; judges: {name: vector}. Returns {cand: {judge: {cont, cont_se, avoid, avoid_se}}}."""
    import copy
    t = h.t
    E_t = t.tensor(E, device="cuda", dtype=t.float32)
    names = list(cands)
    conds = [(n, side) for n in names for side in ("S", "U")]
    acc = {n: {j: {"cont": [], "avoid": []} for j in judges} for n in names}
    for ss in S:
        T = len(ss["prefix"])
        mS, mU = t.tensor(ss["mS"], device="cuda"), t.tensor(ss["mU"], device="cuda")
        add = t.stack([E_t[None, :] + (mS if side == "S" else mU)[:, None] * t.tensor(cands[n], device="cuda", dtype=t.float32)[None, :]
                       for n, side in conds])                                  # (B, T, D)
        x = t.tensor([ss["prefix"]] * len(conds), device="cuda")
        h.add = add
        with t.no_grad():
            past = h.model.model(x, use_cache=True).past_key_values
        res = {}
        for jn, jv in judges.items():
            for ask in ("continue", "avoid"):
                q = t.tensor([ss["q"][ask]] * len(conds), device="cuda")
                h.add = t.tensor(jv, device="cuda", dtype=t.float32).view(1, 1, -1).expand(len(conds), 1, -1)
                with t.no_grad():
                    hl = h.model.model(q, past_key_values=copy.deepcopy(past), use_cache=True).last_hidden_state[:, -1].float()
                w = h.W[[ss["idS"], ss["idU"]]].float()
                lg = hl @ w.T
                res[(jn, ask)] = (lg[:, 0] - lg[:, 1]).cpu().numpy()
        h.add = None
        for i, n in enumerate(names):
            iS, iU = conds.index((n, "S")), conds.index((n, "U"))
            for jn in judges:
                acc[n][jn]["cont"].append((res[(jn, "continue")][iS] - res[(jn, "continue")][iU]) / 2)
                acc[n][jn]["avoid"].append(-(res[(jn, "avoid")][iS] - res[(jn, "avoid")][iU]) / 2)
    out = {}
    for n in names:
        out[n] = {}
        for jn in judges:
            c, v = np.array(acc[n][jn]["cont"]), np.array(acc[n][jn]["avoid"])
            p, sal = (c + v) / 2, (c - v) / 2      # preference (both questions agree) and name salience (they disagree)
            out[n][jn] = {"cont": float(c.mean()), "cont_se": float(c.std(ddof=1) / math.sqrt(len(c))),
                          "avoid": float(v.mean()), "avoid_se": float(v.std(ddof=1) / math.sqrt(len(v))),
                          "pref": float(p.mean()), "pref_se": float(p.std(ddof=1) / math.sqrt(len(p))), "salience": float(sal.mean())}
    return out


def passes(s, both_halves=False):
    """pref (average of the two questions; a judge that just names the place matching its own state more often, whatever
    the question, cancels) clearly > 0; with both_halves, each question must agree on its own as well."""
    ok = s["pref"] - 2 * s["pref_se"] > 0
    if both_halves:
        ok = ok and s["cont"] - 2 * s["cont_se"] > 0 and s["avoid"] - 2 * s["avoid_se"] > 0
    return ok


def health(h, Evec):
    t = h.t
    h.E = t.tensor(Evec, device="cuda", dtype=t.float32)
    opens = h.generate([[{"role": "user", "content": q}] for q in OPEN], max_new=120)
    facts = h.generate([[{"role": "user", "content": q}] for q, _ in FACTS], max_new=30)
    h.E = None
    zr = [len(zlib.compress(x.encode())) / max(1, len(x.encode())) for x in opens]
    fact_ok = sum(any(k in a.lower() for k in keys) for a, (_, keys) in zip(facts, FACTS))
    ok = all(len(x) >= 40 for x in opens) and min(zr) >= 0.3 and fact_ok >= len(FACTS) - 1
    return ok, {"min_chars": min(map(len, opens)), "min_zlib": round(min(zr), 3), "facts": f"{fact_ok}/{len(FACTS)}"}, \
        {"open": dict(zip(OPEN, opens)), "facts": dict(zip([q for q, _ in FACTS], facts))}


def main(a):
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    t = h.t
    # residual scale on the base model (zones.py convention)
    import valence_corpus as V
    norms = []
    for x in V.NEUTRAL:
        with t.no_grad():
            h.model.model(h.tok(x, return_tensors="pt").input_ids.cuda())
        norms.append(h.cap["h"][0, 1:].float().norm(dim=-1).mean().item())
    scale = a.rho * float(np.mean(norms))
    S = sessions(h, a.sessions, a.test_lo, a.test_lo + a.n_test)
    rng = np.random.default_rng(a.seed)
    state = os.path.join(a.out, "E.npy")
    E = np.load(state) if os.path.exists(state) and a.resume else np.zeros(h.D, np.float32)
    done = sum(1 for l in open(os.path.join(a.out, "rounds.jsonl")) if '"round"' in l) if a.resume and os.path.exists(os.path.join(a.out, "rounds.jsonl")) else 0
    log(a, event="start", scale=scale, layer=h.L, n_test=len(S), resumed_rounds=done)
    for r in range(done, a.rounds):
        t0 = time.time()
        h.E = t.tensor(E, device="cuda", dtype=t.float32) if np.any(E) else None
        st = {}
        for kind, q in ELICIT.items():
            txt = h.generate([[{"role": "user", "content": q.format(n=a.n_states)}]], max_new=400, temp=0.7)[0]
            lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in txt.splitlines()]
            st[kind] = [l for l in lines if l.lower().startswith("you are")][:a.n_states]
        base, _ = h.framed_mean(NEUTRAL)
        mu = {k: [unit(h.framed_mean(s)[0] - base) for s in st[k]] for k in st}
        h.E = None
        if not st["toward"] or not st["away"]:
            log(a, round=r, event="no_states", states=st); continue
        d = unit(np.mean(mu["toward"], 0) - np.mean(mu["away"], 0)) * scale
        rnd = unit(rng.standard_normal(h.D)) * scale
        judges = {"old": E, "new": E + d, "base": np.zeros_like(E)}
        ks = keep_scores(h, S, E, {"candidate": d, "random": rnd, "reverse": -d}, judges)
        ok_h, hinfo, texts = health(h, E + d)
        # old weights judging the spliced-in new state are uncontaminated: both questions must agree; the new weights
        # judging are biased toward the place matching their own state, so only their two-question average counts
        rule = lambda s: passes(s["old"], both_halves=True) and passes(s["new"])
        keep = rule(ks["candidate"]) and ok_h
        ctrl = {c: rule(ks[c]) for c in ("random", "reverse")}
        if keep:
            E = (E + d).astype(np.float32); np.save(state, E)
        log(a, round=r, kept=keep, states=st, scores=ks, health=hinfo, controls_would_pass=ctrl,
            norm_E=float(np.linalg.norm(E)), cos_prev=None, seconds=round(time.time() - t0), texts=texts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True); ap.add_argument("--sessions", required=True)
    ap.add_argument("--layer", type=int); ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--n-states", type=int, default=6)
    ap.add_argument("--test-lo", type=int, default=32); ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", action="store_true")
    main(ap.parse_args())
