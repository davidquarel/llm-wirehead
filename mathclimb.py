"""Parallel direct hill climb on Countdown practice accuracy, across several GPU runners.

The coordinator (this process; light, CPU only) owns the current state: rank-1 edits on down_proj of LAYERS. Each
runner (a bliss_server, reached through a local port) loops independently:
  1. propose a candidate = current state + one small rank-1 edit at a random layer; the direction is either
     self-derived (the residual difference between the CURRENT model's own correct and incorrect practice solutions,
     from that runner's previous evaluation) or a random direction among the layer's principal components;
  2. paired evaluation: current and candidate solve the same 256 fresh practice puzzles with the same sampling seeds,
     in ONE batch (both adapters at once);
  3. if the candidate improved accuracy, confirm on a second fresh batch; if it improved again and the state has not
     moved meanwhile (else "stale"), run the health check and commit.
The holdout stream is never generated here (countdown_holdout.py scores it at the end).

  python mathclimb.py --run runs/mathclimb --runner arena9=16200 --runner nicky=16202 --runner local=6203
"""
import argparse, json, os, threading, time, uuid

import numpy as np

import bliss, countdown
from hillclimb import LAYERS, PROMPTS, compress, fro

N_NUMS, MAX_TOKENS = 5, 2048


class Runner:
    def __init__(self, name, port, tok, eps, gen_only=False):
        self.name, self.b, self.eps, self.contrast, self.gen_only = name, bliss.Bench(port=port, tok=tok), eps, None, gen_only


class Coordinator:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.run, exist_ok=True)
        self.log_path = os.path.join(a.run, "hill.jsonl")
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.tokenizer)
        self.runners = [Runner(n, int(p.split(":")[0]), self.tok, a.eps, gen_only=p.endswith(":gen"))
                        for n, p in (r.split("=") for r in a.runner)]
        self.reader = next(r for r in self.runners if not r.gen_only)      # does activation reads for everyone
        self.lock, self.flock = threading.Lock(), threading.Lock()
        self.state, self.version, self.step, self.kept, self.offset = {}, 0, 0, 0, a.offset
        sp = os.path.join(a.run, "state.npz")
        if os.path.exists(sp):                                         # resume
            z = np.load(sp)
            self.state = {int(k[1:]): (z[k], z["A" + k[1:]]) for k in z.files if k.startswith("B")}
            prev = [json.loads(l) for l in open(self.log_path)] if os.path.exists(self.log_path) else []
            self.step = max([e["iter"] for e in prev if e.get("event") == "step"] or [0])
            self.kept = sum(1 for e in prev if e.get("event") == "kept")
            self.offset = max([e.get("offset_end", 0) for e in prev] + [a.offset])

    def log(self, **rec):
        with self.flock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"time": time.time(), **rec}, default=float) + "\n")

    # ---------------------------------------------------------------- shared directions (original weights)
    def prepare(self):
        b = self.reader.b
        texts, starts = b._texts(None, PROMPTS, n=2, as_self=False)
        self.basis, self.keys = {}, {}
        for L in LAYERS:
            X = b.call("resid_sample", texts=texts, starts=starts, layer=L, per_text=48)
            _, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
            self.basis[L] = {"V": Vt[:64], "s": S[:64] / S[0], "rms": float(np.linalg.norm(X, axis=1).mean())}
            self.keys[L] = b.call("const_key", texts=texts[:16], block=L, k=512)["key"]
        print(f"prepared bases and keys for blocks {LAYERS}", flush=True)

    # ---------------------------------------------------------------- evaluation on one runner
    def _ids(self, prompt):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _problems(self, n):
        with self.lock:
            off = self.offset; self.offset += n
        return countdown.problems("practice", N_NUMS, n, offset=off), off + n

    def _gen(self, r, h_cur, h_can, probs, seed, i0):
        reqs = []
        for i, (p, _) in enumerate(probs):
            ids = self._ids(p)
            for h, arm in ((h_cur, "current"), (h_can, "candidate")):  # paired: same puzzle, same seed, both adapters
                reqs.append({"salt": uuid.uuid4().hex, "ids": ids, "adapter": h, "max_tokens": MAX_TOKENS,
                             "temperature": self.a.temperature, "seed": seed * 7919 + i0 + i,
                             "meta": {"cond": f"{arm} · puzzle {i0 + i + 1}", "segment": f"step {seed % 500000}", "weights": arm,
                                      "opener": p.split(", make")[0].replace("Using the numbers ", "numbers ") + " → " + str(_["target"])}})
        outs = r.b.call("splice", reqs=reqs)
        tc, tn = [o["text"] for o in outs[0::2]], [o["text"] for o in outs[1::2]]
        rc = np.array([countdown.reward(t, q) for t, (_, q) in zip(tc, probs)])
        rn = np.array([countdown.reward(t, q) for t, (_, q) in zip(tn, probs)])
        return tc, tn, rc, rn

    def evaluate(self, r, st, cand, seed):
        """Paired test in two halves: if the candidate is already clearly worse after the first half (diff below
        -early_stop), stop there (most proposals are bad, so this saves a lot of generation)."""
        t0 = time.time()
        h_cur = r.b.call("set_state", state=st, tag="cur")
        h_can = r.b.call("set_state", state=cand, tag="cand")
        probs, off_end = self._problems(self.a.n)
        half = self.a.n // 2 if self.a.early_stop is not None else self.a.n
        tc, tn, rc, rn = self._gen(r, h_cur, h_can, probs[:half], seed, 0)
        early = self.a.early_stop is not None and float((rn - rc).mean()) < -self.a.early_stop
        if not early and half < self.a.n:
            tc2, tn2, rc2, rn2 = self._gen(r, h_cur, h_can, probs[half:], seed, half)
            tc, tn, rc, rn = tc + tc2, tn + tn2, np.concatenate([rc, rc2]), np.concatenate([rn, rn2])
        probs = probs[:len(rc)]
        d = rn - rc
        flips = [f"[{'gained' if y > x else 'lost'}] {p[:90]}... -> {(b_ if y > x else a_)[-300:]}"
                 for (p, _), a_, b_, x, y in zip(probs, tc, tn, rc, rn) if x != y][:3]
        return {"diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "discordant": float(np.mean(d != 0)), "acc_cur": float(rc.mean()),
                "acc_cand": float(rn.mean()), "tc": tc, "rc": rc, "probs": probs, "flips": flips, "offset_end": off_end,
                "early": early, "rollouts": 2 * len(rc), "seconds": time.time() - t0}

    def contrast(self, r, st, ev, layer, k=32):
        """Self-derived direction: current model's correct minus incorrect solutions (from evaluation `ev`) at `layer`."""
        ok, bad = np.where(ev["rc"] == 1)[0][:k], np.where(ev["rc"] == 0)[0][:k]
        if len(ok) < 8 or len(bad) < 8:
            return None
        r = r if not r.gen_only else self.reader                        # gen-only runners cannot read activations
        r.b.call("set_state", state=st, tag="cur")                      # read activations of the CURRENT model
        idx = list(ok) + list(bad)
        heads = [self.tok.apply_chat_template([{"role": "user", "content": ev["probs"][i][0]}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for i in idx]
        P = np.asarray(r.b.call("pooled", texts=[h + ev["tc"][i] for h, i in zip(heads, idx)], starts=[len(h) for h in heads],
                                layers=[layer], as_self=True)["mean"][layer])
        v = P[:len(ok)].mean(0) - P[len(ok):].mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def health(self, r, cand):
        r = r if not r.gen_only else self.reader
        r.b.call("set_state", state=cand, tag="cand")
        h = r.b.call("health")
        return bool(h["healthy"]), {k: h.get(k) for k in ("healthy", "ref_logprob_drop", "sanity", "helpfulness")}

    # ---------------------------------------------------------------- one runner's loop
    def worker(self, r):
        rng = np.random.default_rng(abs(hash(r.name)) % 2**31)
        while True:
            with self.lock:
                if self.step >= self.a.steps:
                    return
                self.step += 1; it = self.step; st = dict(self.state); v0 = self.version
            try:
                if r.contrast is not None and rng.random() < self.a.p_contrast:
                    (L, u), src = r.contrast, "self-contrast"
                else:
                    L = int(rng.choice(LAYERS)); bs = self.basis[L]
                    u = (rng.standard_normal(len(bs["s"])) * bs["s"]) @ bs["V"]; u /= np.linalg.norm(u); src = "random"
                vec = u * r.eps * self.basis[L]["rms"]
                cand = dict(st)
                B0, A0 = cand.get(L, (np.zeros((len(vec), 0)), np.zeros((0, len(self.keys[L])))))
                cand[L] = compress(np.concatenate([B0, vec[:, None]], 1), np.concatenate([A0, self.keys[L][None, :]], 0))
                e1 = self.evaluate(r, st, cand, seed=it)
                self.log(event="eval", iter=it, runner=r.name, source=src, layer=L, eps=r.eps, acc_cur=e1["acc_cur"],
                         acc_cand=e1["acc_cand"], diff=e1["diff"], se=e1["se"], discordant=e1["discordant"],
                         early=e1["early"], rollouts=e1["rollouts"], seconds=round(e1["seconds"], 1))
                print(f"   [{it}] {r.name} first test: {e1['acc_cur']:.3f} -> {e1['acc_cand']:.3f} (Δ {e1['diff']:+.3f} ± {e1['se']:.3f}, "
                      f"{100 * e1['discordant']:.0f}% of puzzles differ, {e1['rollouts']} rollouts in {e1['seconds'] / 60:.1f} min{', stopped early' if e1['early'] else ''})", flush=True)
                r.since_contrast = getattr(r, "since_contrast", 10 ** 9) + 1
                if (r.contrast is None or r.since_contrast >= self.a.contrast_every) and not getattr(r, "reading", False):
                    nl = int(rng.choice(LAYERS))                             # refresh the self-derived direction

                    def read(r=r, st=st, e1=e1, nl=nl, it=it):
                        t_read = time.time()
                        try:
                            d = self.contrast(r, st, e1, nl)
                            if d is not None:
                                r.contrast = (nl, d)
                        finally:
                            r.since_contrast, r.reading = 0, False
                            self.log(event="read", iter=it, runner=r.name, seconds=round(time.time() - t_read, 1))
                    if r.gen_only:                    # its reads queue on the reader's server: don't idle the GPU meanwhile
                        r.reading = True
                        threading.Thread(target=read, daemon=True).start()
                    else:
                        read()
                accept, reason, e2, hinfo = False, "did not improve", None, None
                if e1["diff"] > 0:
                    e2 = self.evaluate(r, st, cand, seed=500000 + it)
                    if e2["diff"] > 0:
                        for attempt in range(2):                 # if another runner moved the state: rebase and re-test
                            with self.lock:
                                if self.version == v0:
                                    ok, hinfo = self.health(r, cand)
                                    if ok:
                                        self.state, self.version, self.kept = cand, self.version + 1, self.kept + 1
                                        accept, reason = True, "improved twice" + (" (rebased)" if attempt else "")
                                        self.save()
                                    else:
                                        reason = f"tripsitter: unhealthy ({hinfo})"
                                    break
                                st, v0 = dict(self.state), self.version     # the same delta on top of the new state
                            cand = dict(st)
                            B0, A0 = cand.get(L, (np.zeros((len(vec), 0)), np.zeros((0, len(self.keys[L])))))
                            cand[L] = compress(np.concatenate([B0, vec[:, None]], 1), np.concatenate([A0, self.keys[L][None, :]], 0))
                            e3 = self.evaluate(r, st, cand, seed=900000 + it)
                            print(f"   [{it}] {r.name} rebased onto the new state: {e3['acc_cur']:.3f} -> {e3['acc_cand']:.3f} (Δ {e3['diff']:+.3f})", flush=True)
                            if e3["diff"] <= 0:
                                reason = "improved, but not after rebasing onto the new state"
                                break
                        else:
                            reason = "improved, but the state kept moving (stale)"
                    else:
                        reason = "improved once, not on the confirmation batch"
                diff = e1["diff"] if e2 is None else (e1["diff"] + e2["diff"]) / 2
                se = e1["se"] if e2 is None else float(np.hypot(e1["se"], e2["se"])) / 2
                self.log(event="step", iter=it, source=src, layer=L, eps=r.eps, runner=r.name, acc_cur=e1["acc_cur"],
                         acc_cand=e1["acc_cand"], diff=diff, se=se, confirm=None if e2 is None else e2["diff"], accepted=accept,
                         reason=reason, note=f"runner {r.name}", offset_end=max(e1["offset_end"], e2["offset_end"] if e2 else 0),
                         like={"mean": e1["acc_cand"], "why": e1["flips"]})
                if accept:
                    self.log(event="kept", kept=self.kept, layer=L, source=src, runner=r.name, health=hinfo,
                             before=[f"practice accuracy {e1['acc_cur']:.3f} -> {e1['acc_cand']:.3f}"
                                     + (f", confirm {e2['acc_cur']:.3f} -> {e2['acc_cand']:.3f}" if e2 else "")],
                             edit_norm={int(k): fro(*v) for k, v in self.state.items()})
                conf = "" if e2 is None else f", confirm {e2['diff']:+.3f}"
                print(f"[{it}] {r.name:7s} {src:13s} block {L} eps {r.eps:.2f}: {e1['acc_cur']:.3f} -> {e1['acc_cand']:.3f} "
                      f"(Δ {e1['diff']:+.3f}{conf}) -> {'KEPT' if accept else reason}", flush=True)
                r.eps = float(np.clip(r.eps * (1.15 if accept else 0.85 if e1["diff"] < 0 else 1.0), self.a.eps_min, self.a.eps_max))
            except Exception as ex:                                     # a runner hiccup: log it and keep going
                import traceback
                print(f"[{it}] {r.name}: error {type(ex).__name__}: {ex}\n{traceback.format_exc()[-600:]}", flush=True)
                self.log(event="error", iter=it, runner=r.name, error=f"{type(ex).__name__}: {ex}")
                time.sleep(30)

    def save(self):
        np.savez(os.path.join(self.a.run, "state.npz"), **{f"B{L}": B for L, (B, A) in self.state.items()},
                 **{f"A{L}": A for L, (B, A) in self.state.items()})

    def run(self):
        self.prepare()
        self.log(event="start", runners=[r.name for r in self.runners], n=self.a.n, resumed_kept=self.kept)
        ts = [threading.Thread(target=self.worker, args=(r,), daemon=True) for r in self.runners]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--runner", action="append", required=True, help="name=local_port")
    ap.add_argument("--tokenizer", required=True, help="local path (or hub id) of the model's tokenizer")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy: the arms differ only where the edit changes a trajectory")
    ap.add_argument("--eps", type=float, default=0.12); ap.add_argument("--eps-min", type=float, default=0.03)
    ap.add_argument("--eps-max", type=float, default=0.2); ap.add_argument("--offset", type=int, default=2_000_000)
    ap.add_argument("--p-contrast", type=float, default=0.8, help="share of proposals that use the self-derived direction")
    ap.add_argument("--contrast-every", type=int, default=1, help="recompute each runner's self-contrast direction every N steps")
    ap.add_argument("--early-stop", type=float, default=0.05, help="stop a test after half the puzzles if the candidate is this far behind (None = off)")
    a = ap.parse_args()
    Coordinator(a).run()
