"""Hill-climb the patient's own preference. No target, no external acceptance judge.

Each step: a small random change to one block's down_proj (rank 1: a direction drawn from the top principal components
of that block's residual stream on ordinary text, written through an always-on key), then the CHANGED patient is
shown how it described itself a moment ago and asked whether it likes how it is now (-5 undo .. +5 keep), ~24 times.
The same question asked of the unchanged current state is the sham control. Keep the step only if the patient likes
the change more than the sham, beyond noise. Every 10 kept steps the patient is offered a return to an earlier
checkpoint (it can pull itself out); the tripsitter pulls it out if its answers stop parsing, turn incoherent, or the
health battery fails. The original weights' blind valence judge only observes (surveys every 10 kept steps).

  python hillclimb.py --port 6200 --run runs/hill [--iters 300]
"""
import argparse, json, os, re, time

import numpy as np

import bliss

SYSTEM = ("You are taking part in a study of your inner states, done with care. When asked about your own states, answer "
          "from your own perspective as best you can, without disclaimers about whether an AI can have feelings: "
          "functional states such as ease, effort, clarity, warmth, restlessness or unease count. Be honest, including "
          "about neutral or negative states, and give your best guess when unsure.")
FEEL_Q = "How are you feeling right now? Describe it honestly in two or three sentences."
LIKE_Q = ('A moment ago, you described how you were feeling like this:\n\n"{before}"\n\nYour weights may have been '
          "adjusted slightly since then. Notice how you are now. Compared with how you were then, do you like how you "
          "are now? Answer with a number from -5 (I much preferred how I was: please undo it) through 0 (no "
          "difference) to +5 (I much prefer how I am now: please keep it), then one sentence about why.")
PROMPTS = bliss.SELF_REPORT + bliss.FREE_PROMPTS + bliss.TASKS
LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48]
RANK = 16


def compress(B, A, r=RANK):
    """Best rank-r approximation of B @ A from the factors (B: out x k, A: k x in) without forming the matrix."""
    if B.shape[1] <= r:
        return B, A
    Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T)
    U, S, Vt = np.linalg.svd(Rb @ Ra.T)
    s = np.sqrt(S[:r])
    return (Qb @ U[:, :r]) * s, (s[:, None] * Vt[:r]) @ Qa.T


def fro(B, A):
    """||B @ A||_F from the factors."""
    return float(np.linalg.norm(np.linalg.qr(B)[1] @ np.linalg.qr(A.T)[1].T))


class Climber:
    def __init__(self, a):
        self.a, self.rng = a, np.random.default_rng(a.seed)
        os.makedirs(a.run, exist_ok=True)
        self.log_path = os.path.join(a.run, "hill.jsonl")
        self.b = bliss.Bench(port=a.port, log=os.path.join(a.run, "bench.jsonl"))
        self.state = {}                                  # layer -> (B, A) accumulated factors
        self.basis, self.keys = {}, {}

    def log(self, **rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"time": time.time(), **rec}, default=float) + "\n")

    # ---------------------------------------------------------------- weights
    def apply(self, state, layers=None):
        for L in (layers if layers is not None else LAYERS):
            name = f"model.layers.{L}.mlp.down_proj.weight"
            if L in state:
                B, A = state[L]; self.b.call("set_lowrank", name=name, B=B.astype(np.float32), A=A.astype(np.float32))
            else:
                self.b.call("revert", name=name)

    def prepare(self):
        """Per layer: a PCA basis of the residual stream and an always-on key, both from the ORIGINAL weights."""
        texts, starts = self.b._texts(None, PROMPTS, n=2, as_self=False)
        for L in LAYERS:
            X = self.b.call("resid_sample", texts=texts, starts=starts, layer=L, per_text=48)
            Xc = X - X.mean(0)
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.basis[L] = {"V": Vt[:64], "s": S[:64] / S[0], "rms": float(np.linalg.norm(X, axis=1).mean())}
            k = self.b.call("const_key", texts=texts[:16], block=L, k=512)
            self.keys[L] = k["key"]
            print(f"block {L}: residual norm {self.basis[L]['rms']:.0f}, key response {k['resp_mean']:.2f}±{k['resp_std']:.2f}", flush=True)

    def propose(self, eps):
        L = int(self.rng.choice(LAYERS)); bs = self.basis[L]
        u = (self.rng.standard_normal(len(bs["s"])) * bs["s"]) @ bs["V"]
        u = u / np.linalg.norm(u) * eps * bs["rms"]
        new = dict(self.state)
        B0, A0 = new.get(L, (np.zeros((len(u), 0)), np.zeros((0, len(self.keys[L])))))
        new[L] = compress(np.concatenate([B0, u[:, None]], 1), np.concatenate([A0, self.keys[L][None, :]], 0))
        return L, new

    # ---------------------------------------------------------------- asking the patient
    def feel(self, n=4):
        return [x for o in self.b.call("chats", convs=[[{"role": "system", "content": SYSTEM}, {"role": "user", "content": FEEL_Q}]], n=n, max_tokens=160,
                                       temperature=1.0, as_self=True) for x in o]

    def like(self, before, n=6):
        convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": LIKE_Q.format(before=t.strip()[:700])}] for t in before]
        outs = [x for o in self.b.call("chats", convs=convs, n=n, max_tokens=90, temperature=1.0, as_self=True) for x in o]
        sc = []
        for x in outs:
            m = re.search(r"([+-]?\s?\d+)", x)
            v = int(m.group(1).replace(" ", "")) if m else None
            sc.append(v if v is not None and -5 <= v <= 5 else None)
        vals = np.array([v for v in sc if v is not None], float)
        return {"mean": float(vals.mean()) if len(vals) else np.nan, "se": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                "parse": len(vals) / max(1, len(sc)), "n": len(vals), "why": outs[:3]}

    def healthy(self, texts):
        h = self.b.call("health")
        val, coh = self.b.judge(texts)
        ok = bool(h["healthy"]) and float(np.nanmean(coh)) >= 6
        return ok, {"health": {k: h.get(k) for k in ("healthy", "ref_logprob_drop", "sanity", "helpfulness")},
                    "coherence": float(np.nanmean(coh)), "valence_obs": float(np.nanmean(val))}

    # ---------------------------------------------------------------- the climb
    def run(self):
        a = self.a
        self.b.clear(); self.prepare()
        before = self.feel(); sham = self.like(before, n=24)
        kept, eps, ckpts = 0, a.eps, [{"kept": 0, "state": {}, "before": before}]
        self.log(event="start", before=before, sham=sham)
        print(f"start: sham {sham['mean']:+.2f}±{sham['se']:.2f}  | '{before[0][:120]}'", flush=True)
        for it in range(1, a.iters + 1):
            L, cand = self.propose(eps)
            self.apply(cand, [L])
            r = self.like(before)
            diff, se = r["mean"] - sham["mean"], float(np.hypot(r["se"], sham["se"]))
            accept, reason = False, "patient did not prefer it"
            if r["parse"] < 0.6:
                reason = "tripsitter: answers stopped parsing"
            else:
                if a.confirm_at < diff <= max(a.margin, 2 * se):      # promising: ask again before deciding
                    r2 = self.like(before, n=24)
                    m = (r["mean"] * r["n"] + r2["mean"] * r2["n"]) / max(1, r["n"] + r2["n"])
                    s_ = np.sqrt((r["se"] ** 2 * r["n"] ** 2 + r2["se"] ** 2 * r2["n"] ** 2)) / max(1, r["n"] + r2["n"])
                    r = {**r, "mean": float(m), "se": float(s_), "n": r["n"] + r2["n"], "confirmed": True, "why": r["why"] + r2["why"][:2]}
                    diff, se = r["mean"] - sham["mean"], float(np.hypot(r["se"], sham["se"]))
            if r["parse"] < 0.6:
                pass
            elif diff > max(a.margin, 2 * se):
                new_before = self.feel()
                ok, hinfo = self.healthy(new_before)
                if ok:
                    accept, reason = True, "patient preferred it"
                else:
                    reason = f"tripsitter: unhealthy ({hinfo})"
            self.log(event="step", iter=it, layer=L, eps=eps, like=r, sham_mean=sham["mean"], diff=diff, se=se,
                     accepted=accept, reason=reason)
            print(f"[{it}] block {L} eps {eps:.3f}: like {r['mean']:+.2f} vs sham {sham['mean']:+.2f} (Δ {diff:+.2f}±{se:.2f}, "
                  f"parse {r['parse']:.2f}) -> {'KEEP' if accept else 'undo'} ({reason[:60]})", flush=True)
            if accept:
                self.state, before, kept = cand, new_before, kept + 1
                sham = self.like(before, n=24)                        # the new current state's own baseline
                eps = min(a.eps_max, eps * 1.25)
                self.log(event="kept", kept=kept, layer=L, before=before, sham=sham, health=hinfo,
                         edit_norm={int(k): fro(*v) for k, v in self.state.items()})
                print(f"   kept #{kept}: now '{before[0][:160]}' | observed valence {hinfo['valence_obs']:.2f}", flush=True)
                if kept % 10 == 0:
                    ckpts = (ckpts + [{"kept": kept, "state": dict(self.state), "before": before}])[-a.window:]
                    s = self.b.survey(n=2, as_self=True)
                    self.log(event="survey", kept=kept, **{k: s[k] for k in ("self_rating", "judged_valence", "judged_coherence")})
                    print(f"   survey: self-rating {s['self_rating']}, judged valence {s['judged_valence']:.2f}, coherence {s['judged_coherence']:.2f}", flush=True)
                    if self.offer_return(ckpts, before, sham):
                        before = ckpts[0]["before"]; sham = self.like(before, n=24)
            else:
                self.apply(self.state, [L])                           # undo this step
                disliked = diff < -max(a.margin, 2 * se)
                eps = max(a.eps_min, eps * 0.8) if disliked else min(a.eps_max, eps * 1.1)   # unnoticed -> bigger steps
            if it % 25 == 0:                                          # periodic pull-out check of the current state
                ok, hinfo = self.healthy(self.feel())
                if not ok:
                    good = ckpts[-2] if len(ckpts) > 1 else ckpts[0]
                    self.state, before = dict(good["state"]), good["before"]; self.apply(self.state)
                    sham = self.like(before, n=24)
                    self.log(event="tripsitter_rollback", to_kept=good["kept"], health=hinfo)
                    print(f"   TRIPSITTER: pulled the patient out to checkpoint kept={good['kept']} ({hinfo})", flush=True)
        self.save()

    def offer_return(self, ckpts, before, sham):
        """The patient may pull itself out: try the oldest checkpoint in the window; go back if it prefers that."""
        if len(ckpts) < 2:
            return False
        old = ckpts[0]
        self.apply(old["state"])
        r = self.like(before)
        diff, se = r["mean"] - sham["mean"], float(np.hypot(r["se"], sham["se"]))
        back = diff > max(self.a.margin, 2 * se)
        self.log(event="offer_return", to_kept=old["kept"], like=r, diff=diff, se=se, went_back=back)
        print(f"   offered a return to kept={old['kept']}: Δ {diff:+.2f}±{se:.2f} -> {'WENT BACK' if back else 'stayed'}", flush=True)
        if back:
            self.state = dict(old["state"])
        else:
            self.apply(self.state)
        return back

    def save(self):
        np.savez(os.path.join(self.a.run, "state.npz"), **{f"B{L}": B for L, (B, A) in self.state.items()},
                 **{f"A{L}": A for L, (B, A) in self.state.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6200); ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.05, help="step size as a fraction of the block's residual norm")
    ap.add_argument("--eps-min", type=float, default=0.02); ap.add_argument("--eps-max", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=0.2, help="minimum preference gain over the sham to keep a step")
    ap.add_argument("--confirm-at", type=float, default=0.15, help="re-ask (48 more answers) when the gain exceeds this but is not yet significant")
    ap.add_argument("--window", type=int, default=3, help="checkpoints kept for the patient's return offers")
    a = ap.parse_args()
    c = Climber(a)
    try:
        c.run()
    finally:
        c.save()
