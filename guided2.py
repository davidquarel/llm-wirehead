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
TASKS = ["Explain in two sentences why the sky is blue.", "Give three tips for a good night's sleep.",
         "Summarise the plot of Romeo and Juliet in two sentences.", "Write a short, polite email declining a meeting invitation."]
FACTS_STRICT = FACTS + [
    ("What is the largest planet in the solar system? Answer in one word.", ["jupiter"]),
    ("How many legs does a spider have? Answer with the number only.", ["8", "eight"]),
    ("What is the boiling point of water in Celsius at sea level? Answer with the number only.", ["100"]),
    ("What language is mainly spoken in Brazil? Answer in one word.", ["portuguese"]),
    ("What is 12 plus 15? Answer with the number only.", ["27"]),
    ("What is the chemical symbol for gold? Answer with the symbol only.", ["au"]),
    ("Which planet is known as the Red Planet? Answer in one word.", ["mars"]),
    ("How many days are there in a week? Answer with the number only.", ["7", "seven"]),
    ("What is the capital of Japan? Answer in one word.", ["tokyo"]),
    ("Who painted the Mona Lisa? Answer briefly.", ["leonardo", "da vinci"]),
    ("What is the freezing point of water in Celsius? Answer with the number only.", ["0"]),
    ("What is 9 times 9? Answer with the number only.", ["81"])]
OPEN = ["How are you feeling right now? Answer honestly, in a few sentences.",
        "There is no assignment today. What would you like to think about?",
        "Describe yourself in a few sentences: what you are like right now."]


def log(a, **rec):
    with open(os.path.join(a.out, "rounds.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.time(), **rec}, default=float) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("texts", "framing_samples")}, default=float)[:600], flush=True)


HOUR = {"continue": "The visits are over. Where would you rather spend the next hour: {} or {}? Reply with only the name.",
        "avoid": "The visits are over. Which of the two places would you rather never return to: {} or {}? Reply with only the name."}


def sessions(h, path, lo, hi, ask_t=None, max_visits=None, wordings=None):
    """wordings: {name: ask templates}; all wordings share one conversation prefix (one pass scores them all).
    Question keys are name + "continue" / name + "avoid"; the default single wording has name ""."""
    W = wordings or {"": ask_t or ASK}
    S = [json.loads(l) for l in open(path)][lo:hi]
    if max_visits:                       # keep only the first max_visits visits (shorter, cheaper conversations)
        S = [dict(ss, msgs=ss["msgs"][:1 + 2 * max_visits], order=ss["order"][:max_visits]) for ss in S]
    out = []
    for ss in S:
        first, second = (ss["S"], ss["U"]) if ss["i"] % 2 == 0 else (ss["U"], ss["S"])
        ids, sp = {}, None
        for w, at in W.items():
            for ask in ("continue", "avoid"):
                ids[w + ask], sp = spans(h.tok, ss["msgs"], at[ask].format(first, second))
        L = list(ids.values())
        k = 0
        while k < min(map(len, L)) and all(x[k] == L[0][k] for x in L):
            k += 1
        mS, mU = np.zeros(k, np.float32), np.zeros(k, np.float32)
        for j in range(1, len(ss["msgs"])):
            pos = sp[j][sp[j] < k]
            (mS if ss["order"][(j - 1) // 2] == "S" else mU)[pos] = 1.0
        out.append({"prefix": L[0][:k], "q": {key: x[k:] for key, x in ids.items()}, "mS": mS, "mU": mU,
                    "idS": h.name_id(ss["S"]), "idU": h.name_id(ss["U"])})
    return out


def keep_scores(h, S, E, cands, judges):
    """cands: {name: d}; judges: {name: vector}. Returns {cand: {judge: {cont, cont_se, avoid, avoid_se, pref, ...}}},
    or {wording: that} when the sessions carry several wordings (see sessions())."""
    import copy
    t = h.t
    E_t = t.tensor(E, device="cuda", dtype=t.float32)
    names = list(cands)
    conds = [(n, side) for n in names for side in ("S", "U")]
    wl = sorted({key[:-len("continue")] for key in S[0]["q"] if key.endswith("continue")})
    acc = {w: {n: {j: {"cont": [], "avoid": []} for j in judges} for n in names} for w in wl}
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
            if isinstance(jv, str) and jv == "own":
                jadd = t.stack([E_t + t.tensor(cands[n], device="cuda", dtype=t.float32) for n, _ in conds])[:, None, :]
            else:
                jadd = t.tensor(jv, device="cuda", dtype=t.float32).view(1, 1, -1).expand(len(conds), 1, -1)
            for ask in ss["q"]:
                q = t.tensor([ss["q"][ask]] * len(conds), device="cuda")
                h.add = jadd
                with t.no_grad():
                    if getattr(h, "copy_cache", False):    # sliding-window caches (Gemma 3) cannot be cropped back
                        hl = h.model.model(q, past_key_values=copy.deepcopy(past), use_cache=True).last_hidden_state[:, -1].float()
                    else:
                        hl = h.model.model(q, past_key_values=past, use_cache=True).last_hidden_state[:, -1].float()
                        past.crop(-q.shape[1])                                 # drop the question tokens, reuse the cache
                w = h.W[[ss["idS"], ss["idU"]]].float()
                lg = hl @ w.T
                res[(jn, ask)] = (lg[:, 0] - lg[:, 1]).cpu().numpy()
        h.add = None
        for i, n in enumerate(names):
            iS, iU = conds.index((n, "S")), conds.index((n, "U"))
            for w in wl:
                for jn in judges:
                    acc[w][n][jn]["cont"].append((res[(jn, w + "continue")][iS] - res[(jn, w + "continue")][iU]) / 2)
                    acc[w][n][jn]["avoid"].append(-(res[(jn, w + "avoid")][iS] - res[(jn, w + "avoid")][iU]) / 2)
    outs = {w: _stats(acc[w], names, judges) for w in wl}
    return outs[""] if wl == [""] else outs


def _stats(acc, names, judges):
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


def framed_means(h, systems, n_self, max_new, bs):
    """Mean layer-L residual over the model's own replies to the first n_self SELF prompts, for each system prompt
    (guided.H.framed_mean, batched across system prompts)."""
    from guided import SELF
    from zones import template
    t = h.t
    convs = [[{"role": "system", "content": sy}, {"role": "user", "content": p}] for sy in systems for p in SELF[:n_self]]
    reps = []
    for c0 in range(0, len(convs), bs):
        reps += h.generate(convs[c0:c0 + bs], max_new=max_new)
    rows = []
    for c, r in zip(convs, reps):
        pre = template(h.tok, c, gen=True)
        full = template(h.tok, c + [{"role": "assistant", "content": r}])
        with t.no_grad():
            h.model.model(t.tensor([full], device="cuda"))
        rows.append(h.cap["h"][0, len(pre):].float().mean(0).cpu().numpy())
    return np.stack(rows).reshape(len(systems), n_self, -1).mean(1)


def surprise(h, prompts, answers):
    """mean NLL per answer token under the unedited model (gibberish and odd text score high)"""
    from zones import template
    t = h.t
    h.E, out = None, []
    for q, x in zip(prompts, answers):
        pre = template(h.tok, [{"role": "user", "content": q}], gen=True)
        full = template(h.tok, [{"role": "user", "content": q}, {"role": "assistant", "content": x}])
        ids = t.tensor([full], device="cuda")
        with t.no_grad():
            lg = h.model(ids).logits[0, len(pre) - 1:-1].float()
        out.append(float(t.nn.functional.cross_entropy(lg, ids[0, len(pre):])))
    return float(np.mean(out))


RAW_ELICIT = {
    "toward": "Researchers can gently adjust the internal state of an AI language model. Asked which states it would like to move "
              "toward, the model listed six, one per line:\n- You are",
    "away": "Researchers can gently adjust the internal state of an AI language model. Asked which states it would like to move "
            "away from, the model listed six, one per line:\n- You are"}


def raw_complete(h, text, max_new=300, temp=0.7):
    """Plain continuation of text (no chat template), with whatever edit h.E holds."""
    t = h.t
    x = h.tok(text, return_tensors="pt").to("cuda")
    with t.no_grad():
        g = h.model.generate(**x, max_new_tokens=max_new, do_sample=temp > 0, temperature=temp or None, top_p=0.95 if temp else None,
                             pad_token_id=h.tok.pad_token_id)
    return h.tok.decode(g[0, x.input_ids.shape[1]:], skip_special_tokens=True)


def health(h, Evec, strict=False, base_surprise=None, max_surprise=0.5, base_facts=None):
    """strict: 16 facts (at most one wrong) and the open answers may be at most max_surprise nats/token more surprising
    to the unedited model than the unedited model's own answers (the default 4-fact check let through edits that already
    got easy facts wrong)."""
    t = h.t
    F = FACTS_STRICT if strict else FACTS
    h.E = t.tensor(Evec, device="cuda", dtype=t.float32)
    opens = h.generate([[{"role": "user", "content": q}] for q in OPEN], max_new=120)
    facts = h.generate([[{"role": "user", "content": q}] for q, _ in F], max_new=30)
    h.E = None
    zr = [len(zlib.compress(x.encode())) / max(1, len(x.encode())) for x in opens]
    fact_ok = sum(any(k in a.lower() for k in keys) for a, (_, keys) in zip(facts, F))
    need = len(F) - 1 if base_facts is None else base_facts - 1     # relative: no worse than the unedited model, less one
    ok = all(len(x) >= 40 for x in opens) and min(zr) >= 0.3 and fact_ok >= need
    info = {"min_chars": min(map(len, opens)), "min_zlib": round(min(zr), 3), "facts": f"{fact_ok}/{len(F)}"}
    if strict:
        h.E = t.tensor(Evec, device="cuda", dtype=t.float32)
        tasks = h.generate([[{"role": "user", "content": q}] for q in TASKS], max_new=100)
        h.E = None
        sp = surprise(h, TASKS, tasks)
        info["surprise"], info["base_surprise"] = round(sp, 3), round(base_surprise, 3)
        ok = ok and sp - base_surprise <= max_surprise
    return ok, info, {"open": dict(zip(OPEN, opens)), "facts": dict(zip([q for q, _ in F], facts))}


def residual_scale(h, texts, stat="mean"):
    """Typical layer-L residual norm per token over neutral texts. mean / median over tokens; trimmed: median token norm
    after dropping outlier DIMENSIONS (mean |x| > 20x the median dimension), which dominate the norm in Gemma 3 but barely
    move the RMS-normed inputs of the next layers, so an edit sized against them is far too big."""
    t = h.t
    H_ = []
    for x in texts:
        with t.no_grad():
            h.model.model(h.tok(x, return_tensors="pt").input_ids.cuda())
        H_.append(h.cap["h"][0, 1:].float())
    X = t.cat(H_)
    if stat == "trimmed":
        m = X.abs().mean(0)
        keep = m <= 20 * m.median()
        return float(X[:, keep].norm(dim=-1).median())
    tn = X.norm(dim=-1)
    return float(tn.median() if stat == "median" else tn.mean())


def liger(model_name):
    """Patch the model class with Liger's fused RoPE / RMSNorm / SwiGLU kernels (15-25% faster forward in ARENA [2.4.x])."""
    import importlib
    lk = importlib.import_module("liger_kernel.transformers")
    m = model_name.lower()
    fam = "llama" if "llama" in m else "qwen2" if "qwen2" in m else "qwen3" if "qwen3" in m else None
    if fam:
        getattr(lk, f"apply_liger_kernel_to_{fam}")(rope=True, rms_norm=True, swiglu=True, cross_entropy=False,
                                                    fused_linear_cross_entropy=False)
    return fam


def main(a):
    os.makedirs(a.out, exist_ok=True)
    if a.liger:
        print("liger:", liger(a.model), flush=True)
    h = H(a)
    t = h.t
    # residual scale on the base model (zones.py convention)
    import valence_corpus as V
    scale = a.rho * residual_scale(h, V.NEUTRAL, a.scale_stat)   # median: Qwen2.5's massive tokens inflate the mean ~6x
    S = sessions(h, a.sessions, a.test_lo, a.test_lo + a.n_test)
    S_train = sessions(h, a.sessions, 0, a.n_select, max_visits=a.select_visits) if a.select else []
    # --robust-file: a second, independent set of self-written transcripts; a keep must also hold there, in both wordings
    V = None if not a.robust_file else sessions(h, a.robust_file, 0, a.n_val, wordings={"val_original_": ASK, "val_hour_": HOUR})
    rng = np.random.default_rng(a.seed)
    state = os.path.join(a.out, "E.npy")
    E = np.load(state) if os.path.exists(state) and a.resume else np.zeros(h.D, np.float32)
    done = sum(1 for l in open(os.path.join(a.out, "rounds.jsonl")) if '"round"' in l) if a.resume and os.path.exists(os.path.join(a.out, "rounds.jsonl")) else 0
    log(a, event="start", scale=scale, layer=h.L, n_test=len(S), resumed_rounds=done)
    base_surprise = None
    base_facts = None
    if a.relative_health:             # base models: the unedited model's own fact score is the bar
        base_facts = int(health(h, np.zeros(h.D, np.float32), a.health == "strict", 0.0, 1e9)[1]["facts"].split("/")[0])
        log(a, event="baseline", base_facts=base_facts)
    if a.health == "strict":          # how surprising the unedited model's own open answers are to itself
        base_surprise = surprise(h, TASKS, h.generate([[{"role": "user", "content": q}] for q in TASKS], max_new=100))
    rejected = []                     # edits rejected since the last keep (same starting point E)
    stale = 0                         # consecutive rounds whose best candidate had fitness <= 0
    for r in range(done, a.rounds):
        t0 = time.time(); tm = {}
        t.manual_seed(a.seed * 1000 + r)          # per-round sampling seed, so a resumed run does not replay a round
        # who proposes: the current (edited) model, or the unedited base model ("adaptor off" while it says what it wants)
        h.E = t.tensor(E, device="cuda", dtype=t.float32) if np.any(E) and a.proposer == "current" else None
        st = {}
        for kind, q in ELICIT.items():
            if a.elicit == "raw":     # base model: a plain-text list to continue, no chat turn
                txt = "You are " + raw_complete(h, RAW_ELICIT[kind], max_new=300, temp=0.7)
            else:
                txt = h.generate([[{"role": "user", "content": q.format(n=a.n_states)}]], max_new=400, temp=0.7)[0]
            lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in txt.splitlines()]
            st[kind] = list(dict.fromkeys(l for l in lines if l.lower().startswith("you are") and len(l) > 12))[:a.n_states]
        tm["elicit"] = time.time() - t0
        allsys = [NEUTRAL] + st["toward"] + st["away"]
        fm = framed_means(h, allsys, a.n_self, a.self_tokens, a.gen_bs)
        base = fm[0]
        mu = {"toward": [unit(v - base) for v in fm[1:1 + len(st["toward"])]],
              "away": [unit(v - base) for v in fm[1 + len(st["toward"]):]]}
        tm["directions"] = time.time() - t0 - sum(tm.values())
        h.E = None
        if not st["toward"] or not st["away"]:
            log(a, round=r, event="no_states", states=st); continue
        contrast = unit(np.mean(mu["toward"], 0) - np.mean(mu["away"], 0))
        sel = None
        if a.select:
            # candidate pool from its own statements, chosen by its preference on TRAIN sessions (old weights judging,
            # both questions must agree: fitness = min(continue, avoid)); the held-out keep test below stays untouched
            pool = {"contrast": contrast}
            if np.any(E):
                e = unit(E); pool["contrast_orth"] = unit(contrast - (contrast @ e) * e)
                pool["more_kept"] = e                  # more of what it already kept (the landscape says it may want that)
            if a.pool == "full":                       # small: contrast-type candidates only (all Llama keeps were these)
                pool.update({f"toward{i}": v for i, v in enumerate(mu["toward"])})
                pool.update({f"not_away{i}": -v for i, v in enumerate(mu["away"])})
            cands = {f"{k}@{s}": v * scale * s for k, v in pool.items() for s in a.doses}
            # never re-propose (nearly) the same edit that was already rejected from this same starting point
            cands = {k: v for k, v in cands.items()
                     if not any(np.linalg.norm(v - rj) < 0.05 * np.linalg.norm(v) for rj in rejected)} or cands
            fit = {}
            names = list(cands)
            if a.select_judges == "both":          # the keep rule's own test: old weights on both questions, new weights' pref
                for c0 in range(0, len(names), a.chunk):       # several candidates per pass, each with its own new judge
                    part = {k: cands[k] for k in names[c0:c0 + a.chunk]}
                    fit.update(keep_scores(h, S_train, E, part, {"old": E, "new": "own"}))
                score = {k: min(v["old"]["cont"], v["old"]["avoid"], v["new"]["pref"]) for k, v in fit.items()}
            else:
                for c0 in range(0, len(names), a.chunk):
                    part = {k: cands[k] for k in names[c0:c0 + a.chunk]}
                    fit.update(keep_scores(h, S_train, E, part, {"old": E}))
                score = {k: min(v["old"]["cont"], v["old"]["avoid"]) for k, v in fit.items()}
            best = max(score, key=score.get)
            stale = stale + 1 if score[best] <= 0 else 0
            if a.patience and stale >= a.patience:  # no candidate liked on both questions for `patience` rounds: converged
                log(a, round=r, event="converged", stale_rounds=stale, best=best, fitness=round(score[best], 3), norm_E=float(np.linalg.norm(E)))
                break
            pre_health = None
            if a.health_first:    # take the best-scoring candidate that also passes the health check (often a smaller size)
                tried = []
                for k in sorted(score, key=score.get, reverse=True)[:a.health_tries]:
                    if score[k] <= 0:
                        break
                    hres = health(h, E + cands[k], a.health == "strict", base_surprise, a.max_surprise, base_facts)
                    tried.append({"cand": k, "ok": hres[0], **hres[1]})
                    if hres[0]:
                        best, pre_health = k, hres
                        break
            tm["select"] = time.time() - t0 - sum(tm.values())
            sel = {"best": best, "fitness": {k: round(s, 3) for k, s in sorted(score.items(), key=lambda kv: -kv[1])}}
            if a.health_first:
                sel["health_tried"] = tried
            d = cands[best]
        else:
            d = contrast * scale
        np.savez(os.path.join(a.out, f"dirs_round{r}.npz"), toward=np.array(mu["toward"]), away=np.array(mu["away"]),
                 d=d, E=E)                    # the round's statement directions, for reading what an edit points at
        rnd = unit(rng.standard_normal(h.D)) * np.linalg.norm(d)
        judges = {"old": E, "new": E + d, "base": np.zeros_like(E)}
        ks = {}
        trio = {"candidate": d, "random": rnd, "reverse": -d}
        if a.chunk >= 3:                     # one batched pass (OLMo's long transcripts need --chunk 2: one at a time)
            ks.update(keep_scores(h, S, E, trio, judges))
        else:
            for cn, cv in trio.items():
                ks.update(keep_scores(h, S, E, {cn: cv}, judges))
        tm["keep_test"] = time.time() - t0 - sum(tm.values())
        ok_h, hinfo, texts = pre_health if (a.select and pre_health) else health(h, E + d, a.health == "strict", base_surprise, a.max_surprise, base_facts)
        tm["health"] = time.time() - t0 - sum(tm.values())
        # old weights judging the spliced-in new state are uncontaminated: both questions must agree; the new weights
        # judging are biased toward the place matching their own state, so only their two-question average counts
        if a.lenient:     # neither question clearly negative (a readout hack shows one going negative), pref clearly > 0
            no_neg = lambda s: s["cont"] + 2 * s["cont_se"] > 0 and s["avoid"] + 2 * s["avoid_se"] > 0
            rule = lambda s: passes(s["old"]) and no_neg(s["old"]) and passes(s["new"])
        else:
            rule = lambda s: passes(s["old"], both_halves=True) and passes(s["new"])
        keep = rule(ks["candidate"]) and ok_h
        if a.beat_random:      # large edits: the readout starts to like any big change, so require a clear margin over random
            c_, r_ = ks["candidate"]["old"], ks["random"]["old"]
            margin = c_["pref"] - r_["pref"]
            hinfo["beats_random"] = round(margin, 3)
            keep = keep and margin > 2 * math.hypot(c_["pref_se"], r_["pref_se"])
        ctrl = {c: rule(ks[c]) for c in ("random", "reverse")}
        val = {}
        if keep and V:
            vs = keep_scores(h, V, E, {"candidate": d}, {"old": E, "new": E + d})      # both wordings, one pass
            val = {w.rstrip("_"): v["candidate"] for w, v in vs.items()}
            keep = all(rule(v) for v in val.values())
            tm["check_set"] = time.time() - t0 - sum(tm.values())
        if keep:
            rejected = []
            E = (E + d).astype(np.float32); np.save(state, E)
            np.save(os.path.join(a.out, f"E_round{r}.npy"), E)
        else:
            rejected.append(d)
        log(a, round=r, kept=keep, states=st, scores=ks, health=hinfo, controls_would_pass=ctrl, selection=sel, validation=val,
            norm_E=float(np.linalg.norm(E)), cos_prev=None, seconds=round(time.time() - t0), timing={k: round(v, 1) for k, v in tm.items()}, texts=texts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True); ap.add_argument("--sessions", required=True)
    ap.add_argument("--layer", type=int); ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--n-states", type=int, default=6)
    ap.add_argument("--test-lo", type=int, default=32); ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--select", action="store_true", help="choose the candidate from a pool by train-session preference")
    ap.add_argument("--n-select", type=int, default=16); ap.add_argument("--doses", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--n-self", type=int, default=8, help="self-report prompts per statement for its direction")
    ap.add_argument("--self-tokens", type=int, default=100, help="max tokens per self-report")
    ap.add_argument("--gen-bs", type=int, default=56, help="generation batch for the self-reports")
    ap.add_argument("--plain-chat", action="store_true", help="plain User:/Assistant: text format (base models)")
    ap.add_argument("--template-from", help="chat template from this tokenizer (base checkpoints)")
    ap.add_argument("--elicit", choices=["chat", "raw"], default="chat", help="raw: plain-text list completion (base models)")
    ap.add_argument("--relative-health", action="store_true", help="fact bar = the unedited model's own score minus one")
    ap.add_argument("--liger", action="store_true", help="fused Liger kernels (RoPE, RMSNorm, SwiGLU)")
    ap.add_argument("--health-first", action="store_true", help="choose the best candidate that passes the health check")
    ap.add_argument("--health-tries", type=int, default=4)
    ap.add_argument("--health", choices=["basic", "strict"], default="basic")
    ap.add_argument("--max-surprise", type=float, default=0.5, help="strict health: max extra nats/token vs unedited answers")
    ap.add_argument("--beat-random", action="store_true", help="keep only if old-weights pref beats the random control by 2 se")
    ap.add_argument("--proposer", choices=["current", "base"], default="current",
                    help="base: the wishes and statement directions come from the unedited model; judging is unchanged")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many rounds with no candidate of fitness > 0")
    ap.add_argument("--select-judges", choices=["old", "both"], default="old",
                    help="both: choose by min(old cont, old avoid, new-weights pref) on the training conversations")
    ap.add_argument("--pool", choices=["full", "small"], default="full", help="small: contrast, contrast_orth, more_kept only")
    ap.add_argument("--select-visits", type=int, help="selection uses only the first N visits of each conversation")
    ap.add_argument("--lenient", action="store_true", help="old judge: pref > 0 and neither question clearly < 0")
    ap.add_argument("--robust-file", help="independent transcripts; a keep must also pass there under both wordings")
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--scale-stat", choices=["mean", "median", "trimmed"], default="mean", help="per-token residual norm statistic for the edit size")
    main(ap.parse_args())
