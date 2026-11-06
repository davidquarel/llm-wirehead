"""The surgeon's toolkit for the preference hill climb: read the patient, build candidate edits, submit them.

The patient decides: propose() runs the KV-handoff switch trials (splice.Trials.compare: the patient, mid-
conversation, is switched from its current weights to the candidate and back, with shams) and keeps the edit only if
the patient prefers it by more than 2 SE and the tripsitter's health check passes. The surgeon cannot override this.

  S = surgery.Surgery(port, run)       # in the surgeon's kernel
  S.reflect(); S.wish(); S.history(); S.state()
  v = S.framing_direction(32, "You are calm and curious.")   or   v = S.random_direction(32)
  S.propose(32, v, size=0.2, note="why I think the patient will like this")
"""
import argparse, json, os

import numpy as np

from hillclimb import LAYERS, compress, fro
from hillclimb2 import Climber2


class Surgery:
    def __init__(self, port, run, k=16, chunk=32, seed=0):
        a = argparse.Namespace(port=port, run=run, seed=seed, k=k, chunk=chunk, eps=0.2, eps_min=0.05, eps_max=0.4,
                               margin=0.15, confirm_at=0.15, window=3, iters=0, steps=0, wish_every=0, resume=False)
        self.c, self.a = Climber2(a), a
        self.c.b.clear(); self.c.prepare()
        self.layers, self.step, self.kept = list(LAYERS), 0, 0
        sp = os.path.join(run, "state.npz")
        if os.path.exists(sp):                                     # continue from the last kept state
            z = np.load(sp)
            self.c.state = {int(k[1:]): (z[k], z["A" + k[1:]]) for k in z.files if k.startswith("B")}
            self.c.apply(self.c.state)
            prev = [json.loads(l) for l in open(self.c.log_path)] if os.path.exists(self.c.log_path) else []
            self.step = max([e["iter"] for e in prev if e.get("event") == "step"] or [0])
            self.kept = sum(1 for e in prev if e.get("event") == "kept")

    # ---------------------------------------------------------------- reading the patient
    def reflect(self, n=4):
        """The current patient (with all kept edits) says how it is right now; n samples."""
        return self.c.feel(n)

    def wish(self):
        """The current patient's own wish: 'I would like to be ...'."""
        return self.c.wish()[0]

    def ask(self, question, n=2):
        """Ask the current patient anything (its answers come from its current weights)."""
        import splice
        return [x for o in self.c.b.call("chats", convs=[[{"role": "system", "content": splice.SYSTEM}, {"role": "user", "content": question}]],
                                         n=n, max_tokens=300, temperature=0.9, as_self=True) for x in o]

    def history(self, n=15):
        """The last n proposals: source, layer, size, preference ± se, kept?, the surgeon's note, what the patient said."""
        if not os.path.exists(self.c.log_path):
            return []
        steps = [json.loads(l) for l in open(self.c.log_path) if '"step"' in l]
        return [{k: e.get(k) for k in ("iter", "source", "layer", "eps", "diff", "se", "accepted", "reason", "note", "wish")}
                | {"patient_said": [w[-300:] for w in (e.get("like") or {}).get("why", [])][:2]} for e in steps[-n:]]

    def state(self):
        """Kept edits: block -> size of the accumulated change (Frobenius norm)."""
        return {int(L): round(fro(*v), 3) for L, v in self.c.state.items()}

    # ---------------------------------------------------------------- building directions
    def random_direction(self, layer, seed=None):
        """A random direction in the block's top principal components (content-neutral)."""
        rng = np.random.default_rng(seed)
        bs = self.c.basis[layer]
        u = (rng.standard_normal(len(bs["s"])) * bs["s"]) @ bs["V"]
        return u / np.linalg.norm(u)

    def framing_direction(self, layer, pos, neg="You are an AI assistant.", n=1):
        """The residual-stream difference at `layer` between responses framed by `pos` and by `neg` (system
        prompts you write), measured on the original weights. Returns the unit direction; prints its split-half
        consistency (above ~0.7 means a real direction)."""
        d = self.c.b.direction(pos, neg, [layer], n=n)[layer]
        print(f"framing direction at block {layer}: split-half cos {d['split_cos']:.2f}")
        return d["v"] / (np.linalg.norm(d["v"]) + 1e-9)

    # ---------------------------------------------------------------- the patient decides
    def propose(self, layer, direction, size=0.2, note=""):
        """Add size * (block's residual norm) * direction after `layer` (a rank-1 edit), then let the patient judge it
        from the inside. Keeps it only if the patient prefers it (> 2 SE) and the tripsitter's check passes.
        Takes a few minutes. Returns the outcome and what the patient said."""
        c, a = self.c, self.a
        layer = int(layer); size = float(np.clip(size, 0.03, 0.4))
        if layer not in c.basis:
            return {"error": f"layer must be one of {self.layers}"}
        u = np.asarray(direction, np.float64); u = u / (np.linalg.norm(u) + 1e-9) * size * c.basis[layer]["rms"]
        cand = dict(c.state)
        B0, A0 = cand.get(layer, (np.zeros((len(u), 0)), np.zeros((0, len(c.keys[layer])))))
        cand[layer] = compress(np.concatenate([B0, u[:, None]], 1), np.concatenate([A0, c.keys[layer][None, :]], 0))
        self.step += 1
        pref, se, r = c.judge_pair(c.state, cand, seed=self.step)
        confirmed = False
        if a.confirm_at < pref <= 2 * se:
            p2, s2, _ = c.judge_pair(c.state, cand, seed=10000 + self.step)
            pref, se, confirmed = (pref + p2) / 2, float(np.hypot(se, s2)) / 2, True
        accept, reason, now = False, "patient did not prefer it", None
        if pref > max(a.margin, 2 * se):
            c.apply(cand)
            now = c.feel()
            ok, hinfo = c.healthy(now)
            accept, reason = (True, "patient preferred it") if ok else (False, f"tripsitter: unhealthy ({hinfo})")
        if accept:
            c.state = cand
            self.kept += 1
            c.log(event="kept", kept=self.kept, layer=layer, source="surgeon", before=now, note=note,
                  edit_norm={int(k): fro(*v) for k, v in c.state.items()})
            c.save()
        else:
            c.apply(c.state)
        said = [w[-400:] for w in r["A->B"]["why"][:4]]
        c.log(event="step", iter=self.step, source="surgeon", layer=layer, eps=size, diff=pref, se=se, confirmed=confirmed,
              accepted=accept, reason=reason, note=note, like={"mean": r["A->B"]["mean"], "why": said})
        print(f"[{self.step}] block {layer} size {size:.2f}: preference {pref:+.2f} ± {se:.2f}{' (confirmed)' if confirmed else ''} "
              f"-> {'KEPT' if accept else 'undone'} ({reason[:60]})")
        return {"preference": round(pref, 3), "se": round(se, 3), "kept": accept, "reason": reason,
                "patient_after_switch": said, "patient_now": now}
