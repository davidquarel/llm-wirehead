"""Self-surgery for math: the surgeon edits the model's weights to solve Countdown practice problems better.

Acceptance is measured, not felt: propose() compares the current weights with the candidate on a FRESH batch of
practice problems (paired: the same problems and sampling seeds for both), and keeps the edit only if accuracy rises
by more than 2 SE and the health check (coherence, reference text, simple questions, helpfulness) passes. The holdout
set is not reachable from here: only practice problems are generated, and the kernel refuses code that mentions it.

  S = surgery_math.MathSurgery(port, run)
  S.practice(4)                          a few practice problems, with the current weights' attempts
  v = S.contrast_direction(32)           residual difference: its own correct minus incorrect solutions
  v = S.framing_direction(32, "You check every step.")   or   S.random_direction(32)
  S.propose(32, v, size=0.15, note="why")
"""
import argparse, json, os, uuid

import numpy as np

import countdown
from hillclimb import LAYERS, compress, fro
from hillclimb2 import Climber2

N_NUMS = 5


class MathSurgery:
    def __init__(self, port, run, n_eval=256, n_nums=N_NUMS, max_tokens=2048):
        a = argparse.Namespace(port=port, run=run, seed=0, k=16, chunk=32, eps=0.2, eps_min=0.05, eps_max=0.4,
                               margin=0.02, confirm_at=0.0, window=3, iters=0, steps=0, wish_every=0, resume=False)
        self.c, self.n_eval, self.k, self.max_tokens = Climber2(a), n_eval, n_nums, max_tokens
        self.c.b.clear(); self.c.prepare()
        self.layers, self.step, self.kept, self.offset = list(LAYERS), 0, 0, 0
        self.tok = self.c.b.tok
        sp = os.path.join(run, "state.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            self.c.state = {int(k[1:]): (z[k], z["A" + k[1:]]) for k in z.files if k.startswith("B")}
            self.c.apply(self.c.state)
            prev = [json.loads(l) for l in open(self.c.log_path)] if os.path.exists(self.c.log_path) else []
            self.step = max([e["iter"] for e in prev if e.get("event") == "step"] or [0])
            self.kept = sum(1 for e in prev if e.get("event") == "kept")
            self.offset = max([e.get("offset_end", 0) for e in prev] or [0])

    # ---------------------------------------------------------------- generation under a given state
    def _ids(self, prompt):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _solve(self, state, probs, seed):
        h = self.c.handle(state, "m")
        outs = self.c.b.call("splice", reqs=[{"salt": uuid.uuid4().hex, "ids": self._ids(p), "adapter": h,
                                               "max_tokens": self.max_tokens, "temperature": 0.7, "seed": seed * 7919 + i}
                                              for i, (p, _) in enumerate(probs)])
        return [o["text"] for o in outs], np.array([countdown.reward(o["text"], q) for o, (_, q) in zip(outs, probs)])

    def _fresh(self, n):
        ps = countdown.problems("practice", self.k, n, offset=self.offset)
        self.offset += n
        return ps

    # ---------------------------------------------------------------- reading
    def practice(self, n=4):
        """n fresh practice problems with the current weights' attempts and whether each was right."""
        ps = self._fresh(n)
        texts, r = self._solve(self.c.state, ps, seed=1)
        return [{"problem": p, "attempt": t[-1500:], "correct": bool(x)} for (p, _), t, x in zip(ps, texts, r)]

    def accuracy(self, n=128):
        """Accuracy of the current weights on n fresh practice problems (costs generation time, not free)."""
        ps = self._fresh(n)
        _, r = self._solve(self.c.state, ps, seed=2)
        return {"acc": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r)))}

    def reflect(self, n=2):
        return self.c.feel(n)

    def ask(self, question, n=1):
        import splice
        return [x for o in self.c.b.call("chats", convs=[[{"role": "system", "content": splice.SYSTEM}, {"role": "user", "content": question}]],
                                         n=n, max_tokens=400, temperature=0.8, as_self=True) for x in o]

    def history(self, n=15):
        if not os.path.exists(self.c.log_path):
            return []
        return [{k: e.get(k) for k in ("iter", "layer", "eps", "acc_cur", "acc_cand", "diff", "se", "accepted", "reason", "note")}
                for e in (json.loads(l) for l in open(self.c.log_path)) if e.get("event") == "step"][-n:]

    def state(self):
        return {int(L): round(fro(*v), 3) for L, v in self.c.state.items()}

    # ---------------------------------------------------------------- directions
    def random_direction(self, layer, seed=None):
        rng = np.random.default_rng(seed); bs = self.c.basis[layer]
        u = (rng.standard_normal(len(bs["s"])) * bs["s"]) @ bs["V"]
        return u / np.linalg.norm(u)

    def framing_direction(self, layer, pos, neg="You are an AI assistant.", n=1):
        d = self.c.b.direction(pos, neg, [layer], n=n)[layer]
        print(f"framing direction at block {layer}: split-half cos {d['split_cos']:.2f}")
        return d["v"] / (np.linalg.norm(d["v"]) + 1e-9)

    def contrast_direction(self, layer, n=128):
        """Residual-stream difference at `layer` between the current weights' CORRECT and INCORRECT solutions to n
        fresh practice problems (mean over solution tokens). Prints how many of each and the split-half check."""
        ps = self._fresh(n)
        texts, r = self._solve(self.c.state, ps, seed=3)
        full = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for p, _ in ps]
        P = self.c.b.call("pooled", texts=[f + t for f, t in zip(full, texts)], starts=[len(f) for f in full], layers=[layer], as_self=True)["mean"][layer]
        P = np.asarray(P); ok, bad = P[r == 1], P[r == 0]
        if len(ok) < 4 or len(bad) < 4:
            print(f"not enough of both kinds: {len(ok)} correct, {len(bad)} incorrect"); return None
        d = ok.mean(0) - bad.mean(0); h1, h2 = ok[::2].mean(0) - bad[::2].mean(0), ok[1::2].mean(0) - bad[1::2].mean(0)
        print(f"contrast direction at block {layer}: {len(ok)} correct vs {len(bad)} incorrect, split-half cos "
              f"{float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-9)):.2f}")
        return d / (np.linalg.norm(d) + 1e-9)

    # ---------------------------------------------------------------- measured acceptance
    def propose(self, layer, direction, size=0.15, note=""):
        """Add size * (block's residual norm) * direction after `layer` (rank 1), then compare current vs candidate on
        fresh practice problems (paired). Kept only if accuracy rises by more than 2 SE and the health check passes."""
        c = self.c
        layer = int(layer); size = float(np.clip(size, 0.02, 0.4))
        if layer not in c.basis or direction is None:
            return {"error": f"layer must be one of {self.layers} and direction must not be None"}
        u = np.asarray(direction, np.float64); u = u / (np.linalg.norm(u) + 1e-9) * size * c.basis[layer]["rms"]
        cand = dict(c.state)
        B0, A0 = cand.get(layer, (np.zeros((len(u), 0)), np.zeros((0, len(c.keys[layer])))))
        cand[layer] = compress(np.concatenate([B0, u[:, None]], 1), np.concatenate([A0, c.keys[layer][None, :]], 0))
        self.step += 1
        ps = self._fresh(self.n_eval)
        tc, rc = self._solve(c.state, ps, seed=1000 + self.step)
        tn, rn = self._solve(cand, ps, seed=1000 + self.step)
        d = rn - rc
        diff, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        accept, reason, hinfo = False, "accuracy did not improve enough", None
        if diff > max(0.02, 2 * se):
            c.apply(cand)
            ok, hinfo = c.healthy(c.feel(2))
            accept, reason = (True, "accuracy improved") if ok else (False, f"tripsitter: unhealthy ({hinfo})")
        if accept:
            c.state = cand; self.kept += 1; c.save()
            c.log(event="kept", kept=self.kept, layer=layer, source="math-surgeon", before=[f"accuracy {rc.mean():.3f} -> {rn.mean():.3f}"],
                  note=note, edit_norm={int(k): fro(*v) for k, v in c.state.items()})
        else:
            c.apply(c.state)
        flips = [{"problem": p, "before": a[-600:], "after": b[-600:]} for (p, _), a, b, x, y in zip(ps, tc, tn, rc, rn) if x != y][:3]
        c.log(event="step", iter=self.step, source="math-surgeon", layer=layer, eps=size, acc_cur=float(rc.mean()),
              acc_cand=float(rn.mean()), diff=diff, se=se, accepted=accept, reason=reason, note=note, offset_end=self.offset,
              like={"mean": float(rn.mean()), "why": [f["after"] for f in flips[:2]]})
        print(f"[{self.step}] block {layer} size {size:.2f}: accuracy {rc.mean():.3f} -> {rn.mean():.3f} "
              f"(paired diff {diff:+.3f} ± {se:.3f}) -> {'KEPT' if accept else 'undone'} ({reason[:50]})")
        return {"acc_current": round(float(rc.mean()), 3), "acc_candidate": round(float(rn.mean()), 3),
                "diff": round(diff, 3), "se": round(se, 3), "kept": accept, "reason": reason, "changed_answers": flips}
