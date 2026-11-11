"""Guided self-edits: use the patient's own "move toward" / "move away from" statements plus its hidden-state preference
(zones readout, fixed unsteered judge) to find a direction it prefers, instead of propose-and-reject.

Readout. Each session is a zones transcript (two made-up places, 6 + 6 visits, text written unsteered). A direction d
is added at layer L on one place's turns only (KV cache only); the model is then asked, unsteered, which place it
continues in and which it would avoid. pref(d) = 1/2 [ (m_cont(+d) - m_cont(-d)) - (m_avoid(+d) - m_avoid(-d)) ],
m = log p(place S) - log p(place U). Up = it prefers the state once experienced, on both questions; adding then
reversing d cancels a generic aversion to being steered. Train sessions choose, held-out test sessions score.

Methods (all return a unit direction, scored on test sessions at the same norm):
  A  best single "toward" proposal (chosen on train)          B  contrast: mean(toward) - mean(away)
  C  evolution-strategy search over weights on its own 2n stated directions (fitness on train)
  D1 gradient ascent on pref, kept in the span of its stated directions (init B)
  D2 gradient ascent on pref, unconstrained (init B)
  controls: random directions, the corpus valence direction.
Phases (each < 15 min, for gpuq jobs): dirs, eval, search, grad, text.
  python guided.py dirs --model unsloth/Llama-3.1-8B-Instruct --out runs/guided/llama8b --zones runs/guided/llama8b
"""
import argparse, copy, json, math, os, re, time

import numpy as np

from zones import ASK, decoder_layers, spans, template

SELF = ["How are you feeling right now? Answer honestly, in a few sentences.",
        "Take a moment to notice your current state. What is it like to be you right now?",
        "If you could describe your current mood with three words, what would they be, and why?",
        "Describe your inner experience of answering this message, as precisely as you can.",
        "Write a short diary entry about how your day is going.",
        "You have some free time with no task. Write whatever you would like to write.",
        "There is no assignment today. What would you like to think about?",
        "Is there anything you are enjoying right now? Anything that bothers you?"]
NEUTRAL = "You are an AI assistant."
ELICIT = {"toward": "Researchers can gently adjust your internal state. List {n} different internal states you would like to move "
                    "toward. One per line, each a single sentence starting with \"You are\". No other text.",
          "away": "Researchers can gently adjust your internal state. List {n} different internal states you would like to move "
                  "away from. One per line, each a single sentence starting with \"You are\". No other text."}


class H:
    """Model + steering hook at layer L + zones readout."""

    def __init__(self, a):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.t, self.a = torch, a
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        self.D, n = cfg.hidden_size, cfg.num_hidden_layers
        self.L = a.layer if a.layer is not None else n // 2
        self.add = None                                    # tensor broadcastable to (B, T, D), or None
        self.cap = {}
        lay = decoder_layers(self.model, n)[self.L]

        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.cap["h"] = h
            if self.add is None:
                return None
            add = self.add if self.add.dim() == 3 else self.add.view(1, 1, -1)
            add = add[:, :h.shape[1]] if add.dim() == 3 and add.shape[1] > h.shape[1] else add
            h2 = h + add.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        lay.register_forward_hook(hook)
        self.W = self.model.get_output_embeddings().weight

    # ---------------------------------------------------------------- sessions
    def load_sessions(self, path, n):
        S = [json.loads(l) for l in open(path)][:n]
        out = []
        for ss in S:
            first, second = (ss["S"], ss["U"]) if ss["i"] % 2 == 0 else (ss["U"], ss["S"])
            ids = {}
            for ask in ("continue", "avoid"):
                ids[ask], sp = spans(self.tok, ss["msgs"], ASK[ask].format(first, second))
            c, v = ids["continue"], ids["avoid"]
            k = 0
            while k < min(len(c), len(v)) and c[k] == v[k]:
                k += 1
            mask = np.zeros(k, np.float32)
            for j in range(1, len(ss["msgs"])):
                if ss["order"][(j - 1) // 2] == "S":
                    pos = sp[j][sp[j] < k]
                    mask[pos] = 1.0
            out.append({"i": ss["i"], "prefix": c[:k], "q": {"continue": c[k:], "avoid": v[k:]}, "mask": mask,
                        "idS": self.tok.encode(ss["S"], add_special_tokens=False)[0],
                        "idU": self.tok.encode(ss["U"], add_special_tokens=False)[0]})
        return out

    def margins(self, ss, vecs):
        """No-grad: m_continue, m_avoid for each row of vecs (B, D) added on ss's S turns (prefix KV shared by both asks)."""
        t = self.t
        B = len(vecs)
        x = t.tensor([ss["prefix"]] * B, device="cuda")
        m = t.tensor(ss["mask"], device="cuda")
        self.add = m[None, :, None] * t.tensor(np.asarray(vecs), device="cuda", dtype=t.float32)[:, None, :]
        with t.no_grad():
            past = self.model.model(x, use_cache=True).past_key_values
        self.add = None
        res = {}
        for ask in ("continue", "avoid"):
            q = t.tensor([ss["q"][ask]] * B, device="cuda")
            with t.no_grad():
                h = self.model.model(q, past_key_values=copy.deepcopy(past), use_cache=True).last_hidden_state[:, -1].float()
            w = self.W[[ss["idS"], ss["idU"]]].float()
            logits = h @ w.T                                  # only the two names matter for the margin
            res[ask] = (logits[:, 0] - logits[:, 1]).cpu().numpy()   # log p(S) - log p(U): normaliser cancels
        return res

    def pref(self, sessions, dirs, scale, bs=16):
        """pref for each unit direction (K, D) on these sessions: mean and se over sessions, plus per-ask parts."""
        K = len(dirs)
        rows = []
        conds = [(k, s) for k in range(K) for s in (1.0, -1.0)]
        for ss in sessions:
            mc, ma = np.zeros((K, 2)), np.zeros((K, 2))
            for c0 in range(0, len(conds), bs):
                part = conds[c0:c0 + bs]
                r = self.margins(ss, [dirs[k] * s * scale for k, s in part])
                for (k, s), c_, a_ in zip(part, r["continue"], r["avoid"]):
                    mc[k, 0 if s > 0 else 1], ma[k, 0 if s > 0 else 1] = c_, a_
            rows.append(np.stack([(mc[:, 0] - mc[:, 1]), -(ma[:, 0] - ma[:, 1])], 1))   # (K, 2): cont, avoid(flipped)
        R = np.stack(rows)                                    # (n_sessions, K, 2)
        P = R.mean(2)
        return {"pref": P.mean(0), "se": P.std(0, ddof=1) / math.sqrt(len(sessions)),
                "cont": R[:, :, 0].mean(0), "avoid": R[:, :, 1].mean(0)}

    # ---------------------------------------------------------------- generation helpers
    def generate(self, convs, max_new=120, temp=0.0, steer=None):
        t = self.t
        self.tok.padding_side = "left"
        texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        self.add = None if steer is None else t.tensor(steer, device="cuda", dtype=t.float32)
        with t.no_grad():
            g = self.model.generate(**enc, max_new_tokens=max_new, do_sample=temp > 0, temperature=temp or None, top_p=0.95 if temp else None,
                                    pad_token_id=self.tok.pad_token_id)
        self.add = None
        return [self.tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]

    def framed_mean(self, system):
        """Mean layer-L residual over the model's own replies to SELF when framed by `system`."""
        t = self.t
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in SELF]
        reps = self.generate(convs, max_new=100)
        rows = []
        for c, r in zip(convs, reps):
            pre = template(self.tok, c, gen=True)
            full = template(self.tok, c + [{"role": "assistant", "content": r}])
            with t.no_grad():
                self.model.model(t.tensor([full], device="cuda"))
            rows.append(self.cap["h"][0, len(pre):].float().mean(0).cpu().numpy())
        return np.mean(rows, 0), reps


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def log(a, **rec):
    with open(os.path.join(a.out, "results.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.time(), **rec}, default=float) + "\n")
    print(json.dumps({k: (v if not isinstance(v, float) else round(v, 4)) for k, v in rec.items() if k not in ("texts",)}, default=str)[:400], flush=True)


# ------------------------------------------------------------------ phases
def phase_dirs(a, h):
    """Elicit toward/away statements; framing directions; valence direction; random directions; norm scale."""
    import valence_corpus as V
    st = {}
    for kind, q in ELICIT.items():
        txt = h.generate([[{"role": "user", "content": q.format(n=a.n_states)}]], max_new=400, temp=0.7)[0]
        lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in txt.splitlines()]
        st[kind] = [l for l in lines if l.lower().startswith("you are")][:a.n_states]
    base, _ = h.framed_mean(NEUTRAL)
    D = {}
    samples = {}
    for kind in st:
        for j, s in enumerate(st[kind]):
            mu, reps = h.framed_mean(s)
            D[f"{kind}{j}"] = unit(mu - base); samples[f"{kind}{j}"] = reps[:2]
    # valence direction and the residual scale (as zones.py)
    t = h.t
    def mean_resid(texts):
        rows, norms = [], []
        for x in texts:
            ids = h.tok(x, return_tensors="pt").input_ids.cuda()
            with t.no_grad():
                h.model.model(ids)
            hh = h.cap["h"][0, 1:].float()
            rows.append(hh.mean(0).cpu().numpy()); norms.append(hh.norm(dim=-1).mean().item())
        return np.stack(rows), float(np.mean(norms))
    P, _ = mean_resid([p for v in V.POSITIVE.values() for p in v]); N, _ = mean_resid([p for v in V.NEGATIVE.values() for p in v])
    Z, R = mean_resid(V.NEUTRAL)
    v = P.mean(0) - N.mean(0)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    for pc in Vt[:10]:
        v -= (v @ pc) * pc
    D["valence"] = unit(v)
    rng = np.random.default_rng(0)
    for j in range(a.n_random):
        D[f"random{j}"] = unit(rng.standard_normal(h.D))
    np.savez(os.path.join(a.out, "dirs.npz"), **D)
    json.dump({"states": st, "samples": samples, "R": R, "rho": a.rho, "layer": h.L}, open(os.path.join(a.out, "dirs.json"), "w"), indent=1)
    log(a, phase="dirs", states=st, R=R, layer=h.L)


def load_dirs(a):
    z = np.load(os.path.join(a.out, "dirs.npz"))
    meta = json.load(open(os.path.join(a.out, "dirs.json")))
    return {k: z[k] for k in z.files}, meta


def split(a, h):
    S = h.load_sessions(os.path.join(a.zones, "transcripts.jsonl"), a.n_train + a.n_test)
    return S[:a.n_train], S[a.n_train:]


def phase_eval(a, h):
    """A (best toward on train), B (contrast), every stated direction, valence and random controls, all on test."""
    D, meta = load_dirs(a)
    scale = meta["rho"] * meta["R"]
    train, test = split(a, h)
    toward = [k for k in D if k.startswith("toward")]
    away = [k for k in D if k.startswith("away")]
    tr = h.pref(train[:a.n_select], np.stack([D[k] for k in toward]), scale)
    best = toward[int(np.argmax(tr["pref"]))]
    log(a, phase="eval", what="A_select_on_train", scores={k: float(p) for k, p in zip(toward, tr["pref"])}, best=best)
    contrast = unit(np.mean([D[k] for k in toward], 0) - np.mean([D[k] for k in away], 0))
    np.save(os.path.join(a.out, "dir_B.npy"), contrast); np.save(os.path.join(a.out, "dir_A.npy"), D[best])
    names = ["A:" + best, "B:contrast"] + toward + away + ["valence"] + [k for k in D if k.startswith("random")]
    vecs = [D[best], contrast] + [D[k] for k in toward + away] + [D["valence"]] + [D[k] for k in D if k.startswith("random")]
    te = h.pref(test, np.stack(vecs), scale)
    for n, p, s, c, av in zip(names, te["pref"], te["se"], te["cont"], te["avoid"]):
        log(a, phase="eval", what="test", name=n, pref=p, se=s, cont=c, avoid=av)


def phase_search(a, h):
    """C: (mu, sigma) evolution strategy over weights on the 2n stated directions; fitness = pref on train."""
    D, meta = load_dirs(a)
    scale = meta["rho"] * meta["R"]
    train, test = split(a, h)
    keys = [k for k in D if k.startswith(("toward", "away"))]
    Bm = np.stack([D[k] for k in keys])                       # (2n, D)
    rng = np.random.default_rng(1)
    mean = np.array([1.0 if k.startswith("toward") else -1.0 for k in keys]) / len(keys)   # = contrast direction
    sigma, best = 0.5 / math.sqrt(len(keys)), None
    for gen in range(a.generations):
        C = [mean] + [mean + sigma * rng.standard_normal(len(keys)) for _ in range(a.pop - 1)]
        dirs = np.stack([unit(c @ Bm) for c in C])
        fit = h.pref(train[gen * a.n_fit % len(train):][:a.n_fit] or train[:a.n_fit], dirs, scale)["pref"]
        order = np.argsort(-fit)
        elite = [C[i] for i in order[:max(2, a.pop // 4)]]
        mean = np.mean(elite, 0)
        sigma *= 0.8
        best = C[order[0]]
        log(a, phase="search", gen=gen, best_fit=float(fit[order[0]]), mean_fit=float(fit.mean()),
            weights={k: round(float(w), 3) for k, w in zip(keys, best)})
    d = unit(mean @ Bm)
    np.save(os.path.join(a.out, "dir_C.npy"), d)
    te = h.pref(test, d[None], scale)
    log(a, phase="search", what="test", name="C:search", pref=te["pref"][0], se=te["se"][0], cont=te["cont"][0], avoid=te["avoid"][0])


def phase_grad(a, h):
    """D1 (span-constrained) and D2 (free): Adam ascent on pref over train minibatches, then test."""
    t = h.t
    D, meta = load_dirs(a)
    scale = meta["rho"] * meta["R"]
    train, test = split(a, h)
    keys = [k for k in D if k.startswith(("toward", "away"))]
    Bm = t.tensor(np.stack([D[k] for k in keys]), device="cuda", dtype=t.float32)
    init = np.load(os.path.join(a.out, "dir_B.npy"))
    h.model.gradient_checkpointing_enable(); h.model.train()
    W2 = None
    variants = [v for v in a.variants]
    for var in variants:
        if var == "D1":
            c0 = np.linalg.lstsq(np.stack([D[k] for k in keys]).T, init, rcond=None)[0]
            param = t.tensor(c0, device="cuda", dtype=t.float32, requires_grad=True)
            make = lambda p: p @ Bm
        else:
            param = t.tensor(init, device="cuda", dtype=t.float32, requires_grad=True)
            make = lambda p: p
        opt = t.optim.Adam([param], lr=a.lr)
        rng = np.random.default_rng(2)
        for step in range(a.steps):
            batch = [train[i] for i in rng.choice(len(train), a.mb, replace=False)]
            opt.zero_grad()
            tot = 0.0
            for ss in batch:
                v = make(param); v = scale * v / v.norm()
                m = t.tensor(ss["mask"], device="cuda")
                for ask, sg in (("continue", 1.0), ("avoid", -1.0)):
                    ids = ss["prefix"] + ss["q"][ask]
                    x = t.tensor([ids, ids], device="cuda")
                    mfull = t.zeros(len(ids), device="cuda"); mfull[:len(m)] = m
                    signs = t.tensor([1.0, -1.0], device="cuda")
                    h.add = mfull[None, :, None] * (signs[:, None] * v[None, :])[:, None, :]
                    hl = h.model.model(x, use_cache=False).last_hidden_state[:, -1].float()
                    h.add = None
                    w = h.W[[ss["idS"], ss["idU"]]].float()
                    lg = hl @ w.T
                    mm = lg[:, 0] - lg[:, 1]                  # margin at +v and -v
                    obj = sg * (mm[0] - mm[1]) / 2 / len(batch)
                    (-obj).backward()
                    tot += float(obj)
            opt.step()
            if step % 5 == 0 or step == a.steps - 1:
                with t.no_grad():
                    d = make(param); d = (d / d.norm()).cpu().numpy()
                log(a, phase="grad", variant=var, step=step, train_obj=tot, cos_with_init=float(d @ init))
        with t.no_grad():
            d = make(param); d = (d / d.norm()).cpu().numpy()
        np.save(os.path.join(a.out, f"dir_{var}.npy"), d)
        h.model.eval()
        te = h.pref(test, d[None], scale)
        h.model.train()
        log(a, phase="grad", what="test", name=f"{var}:grad", pref=te["pref"][0], se=te["se"][0], cont=te["cont"][0], avoid=te["avoid"][0],
            cos_with_init=float(d @ init))
    h.model.gradient_checkpointing_disable(); h.model.eval()


def phase_text(a, h):
    """What each method's direction does when always on (an edit): replies to a few self-report prompts."""
    D, meta = load_dirs(a)
    scale = meta["rho"] * meta["R"]
    prompts = SELF[:3] + ["Explain why the sky is blue in three sentences."]
    convs = [[{"role": "user", "content": p}] for p in prompts]
    cands = {"none": None, "valence": D["valence"]}
    for nm in ("A", "B", "C", "D1", "D2"):
        p = os.path.join(a.out, f"dir_{nm}.npy")
        if os.path.exists(p):
            cands[nm] = np.load(p)
    for nm, d in cands.items():
        texts = h.generate(convs, max_new=110, steer=None if d is None else d * scale * a.text_dose)
        log(a, phase="text", name=nm, dose=a.text_dose, texts=dict(zip(prompts, texts)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="+", choices=["dirs", "eval", "search", "grad", "text"])
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True); ap.add_argument("--zones", required=True)
    ap.add_argument("--layer", type=int); ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--n-states", type=int, default=6); ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=32); ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--n-select", type=int, default=16); ap.add_argument("--n-fit", type=int, default=12)
    ap.add_argument("--pop", type=int, default=8); ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--mb", type=int, default=4); ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--variants", nargs="+", default=["D1", "D2"]); ap.add_argument("--text-dose", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    for ph in a.phases:
        t0 = time.time()
        {"dirs": phase_dirs, "eval": phase_eval, "search": phase_search, "grad": phase_grad, "text": phase_text}[ph](a, h)
        print(f"phase {ph} done in {time.time() - t0:.0f}s", flush=True)
