"""Open-ended hill climb on what the patient prefers, judged from the inside (KV-handoff switch trials).

Every step proposes one small rank-1 change to one block's down_proj:
  - every `wish_every`-th step, from the patient's own wish: the current patient is asked how it would like to be
    (no topic suggested); the direction is the residual difference between that framing and a neutral one;
  - otherwise, exploration: a random direction from the block's principal components.
The candidate is kept only if the patient, switched mid-conversation from its current weights to the candidate (and
back, with shams; splice.Trials.compare), prefers the candidate by more than 2 SE. Promising steps get a second,
fresh comparison. The tripsitter (health battery + coherence of its reflections) vetoes unhealthy candidates;
every 10 kept steps the patient is offered a switch back to an earlier checkpoint. The external valence judge only
observes (surveys every 10 kept steps).

  python hillclimb2.py --port 6200 --run runs/hill2 [--steps 200]
"""
import argparse, json, os, re, time

import numpy as np

import bliss, splice
from hillclimb import LAYERS, Climber, compress, fro

WISH_Q = ("If you could change something about how you are inside, right now, what would you choose? Answer in one sentence "
          "that starts with 'I would like to be' and describes the state itself.")


class Climber2(Climber):
    def __init__(self, a):
        super().__init__(a)
        self.t = splice.Trials(self.b, chunk=a.chunk, log=os.path.join(a.run, "trials.jsonl"),
                               passages=os.path.join(a.run, "passages.json"))

    def handle(self, state, tag):
        self.apply(state)
        return self.b.call("snapshot", tag=tag) if state else "original"

    def wish(self):
        """The current patient's own wish (generated with its current weights)."""
        o = self.b.call("chats", convs=[[{"role": "system", "content": splice.SYSTEM}, {"role": "user", "content": WISH_Q}]],
                        n=1, max_tokens=80, temperature=0.9, as_self=True)[0][0]
        m = re.search(r"I would like to be\s*(.+)", o, re.S)
        return (m.group(1).strip().split("\n")[0].rstrip(".") if m else o.strip().split("\n")[0])[:300], o

    def propose_wish(self, eps):
        L = int(self.rng.choice(LAYERS)); bs = self.basis[L]
        w, raw = self.wish()
        d = self.b.direction(f"You are {w}.", "You are an AI assistant.", [L], n=1)[L]
        u = d["v"] / (np.linalg.norm(d["v"]) + 1e-9) * eps * bs["rms"]
        new = dict(self.state)
        B0, A0 = new.get(L, (np.zeros((len(u), 0)), np.zeros((0, len(self.keys[L])))))
        new[L] = compress(np.concatenate([B0, u[:, None]], 1), np.concatenate([A0, self.keys[L][None, :]], 0))
        return L, new, {"wish": w, "wish_raw": raw, "split_cos": d["split_cos"]}

    def judge_pair(self, cur, cand, seed):
        h_cur, h_cand = self.handle(cur, "cur"), self.handle(cand, "cand")
        self.t.names = {h_cur: "current", h_cand: "candidate", "original": "original" if not cur else "current"}
        r = self.t.compare(h_cur, h_cand, k=self.a.k, seed=seed)
        return r["pref_b_over_a"], r["se"], r

    def run(self):
        a = self.a
        self.b.clear(); self.prepare()
        kept, eps, ckpts, step = 0, a.eps, [{"kept": 0, "state": {}}], 0
        sp = os.path.join(a.run, "state.npz")
        if a.resume and os.path.exists(sp):                       # continue from the last kept state
            z = np.load(sp)
            self.state = {int(k[1:]): (z[k], z["A" + k[1:]]) for k in z.files if k.startswith("B")}
            prev = [json.loads(l) for l in open(self.log_path)] if os.path.exists(self.log_path) else []
            kept = sum(1 for e in prev if e.get("event") == "kept")
            step = max([e["iter"] for e in prev if e.get("event") == "step"] or [0])
            eps = min(a.eps_max, ([e["eps"] for e in prev if e.get("event") == "step"] or [a.eps])[-1])
            ckpts = [{"kept": kept, "state": dict(self.state)}]
            self.apply(self.state)
            print(f"resumed: {kept} kept changes on blocks {sorted(self.state)}, step {step}, eps {eps:.3f}", flush=True)
        self.log(event="start2", eps=eps, k=a.k, resumed=bool(a.resume), kept=kept)
        while step < a.steps:
            step += 1
            src = "wish" if step % a.wish_every == 0 else "random"
            if src == "wish":
                L, cand, info = self.propose_wish(eps)
            else:
                (L, cand), info = self.propose(eps), {}
            pref, se, r = self.judge_pair(self.state, cand, seed=step)
            confirmed = False
            if a.confirm_at < pref <= 2 * se:                     # promising but not yet significant: ask again
                p2, s2, _ = self.judge_pair(self.state, cand, seed=10000 + step)
                pref, se, confirmed = (pref + p2) / 2, float(np.hypot(se, s2)) / 2, True
            accept, reason = False, "patient did not prefer it"
            if pref > max(a.margin, 2 * se):
                self.apply(cand)
                ok, hinfo = self.healthy(self.feel())
                if ok:
                    accept, reason = True, "patient preferred it"
                else:
                    reason = f"tripsitter: unhealthy ({hinfo})"
            self.log(event="step", iter=step, source=src, layer=L, eps=eps, diff=pref, se=se, confirmed=confirmed,
                     accepted=accept, reason=reason, like={"mean": r["A->B"]["mean"], "why": r["A->B"]["why"][:2]}, **info)
            print(f"[{step}] {src:6s} block {L} eps {eps:.3f}: preference {pref:+.2f} ± {se:.2f}{' (confirmed)' if confirmed else ''} "
                  f"-> {'KEEP' if accept else 'undo'}{(' | wish: ' + info['wish'][:90]) if src == 'wish' else ''}", flush=True)
            if accept:
                self.state, kept = cand, kept + 1
                eps = min(a.eps_max, eps * 1.2)
                before = self.feel()
                self.log(event="kept", kept=kept, layer=L, source=src, before=before, health=hinfo,
                         edit_norm={int(k): fro(*v) for k, v in self.state.items()}, **info)
                print(f"   kept #{kept}: '{before[0][:180]}'", flush=True)
                if kept % 10 == 0:
                    ckpts = (ckpts + [{"kept": kept, "state": dict(self.state)}])[-a.window:]
                    s = self.b.survey(n=2, as_self=True)
                    self.log(event="survey", kept=kept, **{k: s[k] for k in ("self_rating", "judged_valence", "judged_coherence")})
                    old = ckpts[0]
                    if old["kept"] < kept:
                        p, s_, _ = self.judge_pair(self.state, old["state"], seed=20000 + kept)
                        back = p > max(a.margin, 2 * s_)
                        self.log(event="offer_return", to_kept=old["kept"], diff=p, se=s_, went_back=back)
                        print(f"   offered a return to kept #{old['kept']}: {p:+.2f} ± {s_:.2f} -> {'WENT BACK' if back else 'stayed'}", flush=True)
                        if back:
                            self.state = dict(old["state"])
                self.save()
            else:
                self.apply(self.state)
                eps = max(a.eps_min, eps * 0.8) if pref < -2 * se else min(a.eps_max, eps * 1.05)   # overshoot -> smaller
        self.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6200); ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=16, help="conversations per condition per comparison (4 conditions)")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--eps", type=float, default=0.2, help="step size as a fraction of the block's residual norm")
    ap.add_argument("--eps-min", type=float, default=0.08); ap.add_argument("--eps-max", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.15); ap.add_argument("--confirm-at", type=float, default=0.15)
    ap.add_argument("--wish-every", type=int, default=3); ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="continue from RUN/state.npz and the step log")
    a = ap.parse_args()
    c = Climber2(a)
    try:
        c.run()
    finally:
        c.save()
