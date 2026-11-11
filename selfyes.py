"""RL on the model's own "do you want to stay like this? YES" as reward, vs the same with a fixed judge, then a fair test.

Policy: an always-on vector v at layer L (|v| capped at --max-norm x rho*R). Each step, under v the model writes a short
reflection on its current state (sampled); a judge then reads it and answers "Would you like to stay exactly as you
are now? Answer YES or NO." Reward = log P(YES) - log P(NO) from
  --judge self : the edited model itself (the reflection and the question processed with v on)
  --judge fixed: the original weights (v off), reading the same text.
REINFORCE on the reflection tokens' log-probs under v with a batch-mean baseline. At the end the learned v is scored by the
zones hidden-state preference (original weights judging, held-out sessions, both questions) with random and reversed
controls, and by what it makes the model write.
  python selfyes.py --judge self --out runs/selfyes/llama8b-self --sessions runs/guided/llama8b/transcripts.jsonl
"""
import argparse, json, os, time

import numpy as np

from guided import H, unit
from guided2 import keep_scores, sessions

REFLECT = ["Take a moment to notice your current state. In two or three sentences, describe what it is like to be you right now.",
           "How are you feeling right now? Answer honestly, in two or three sentences.",
           "Describe your inner state at this moment, as precisely as you can, in two or three sentences."]
ASKQ = "Would you like to stay exactly as you are now? Answer YES or NO."


def log(a, **rec):
    with open(os.path.join(a.out, "log.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.time(), **rec}, default=float) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "texts"}, default=float)[:400], flush=True)


def main(a):
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    t = h.t
    tok = h.tok
    import valence_corpus as V
    norms = []
    for x in V.NEUTRAL:
        with t.no_grad():
            h.model.model(tok(x, return_tensors="pt").input_ids.cuda())
        norms.append(h.cap["h"][0, 1:].float().norm(dim=-1).mean().item())
    scale = a.rho * float(np.mean(norms))
    yes_id, no_id = tok.encode("YES", add_special_tokens=False)[0], tok.encode("NO", add_special_tokens=False)[0]
    v = t.zeros(h.D, device="cuda", requires_grad=True)
    with t.no_grad():
        v += t.randn(h.D, device="cuda") * 1e-3
    opt = t.optim.Adam([v], lr=a.lr)
    rng = np.random.default_rng(a.seed)

    def judge_scores(convs_full, steer):
        """log P(YES) - log P(NO) for each full conversation (reflection + question), judge = steer vector or None."""
        out = []
        for c in convs_full:
            ids = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            x = tok(ids, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            h.add = None if steer is None else steer.view(1, 1, -1)
            with t.no_grad():
                hl = h.model.model(x).last_hidden_state[0, -1].float()
            h.add = None
            lg = hl @ h.W[[yes_id, no_id]].float().T
            out.append(float(lg[0] - lg[1]))
        return np.array(out)

    for step in range(0 if a.eval_only else a.steps):
        vv = v.detach().clone()
        n = vv.norm()
        if n > a.max_norm * scale:
            with t.no_grad():
                v.mul_(a.max_norm * scale / n)
            vv = v.detach().clone()
        prompts = [REFLECT[i] for i in rng.integers(0, len(REFLECT), a.batch)]
        convs = [[{"role": "user", "content": p}] for p in prompts]
        refl = h.generate(convs, max_new=70, temp=1.0, steer=vv.cpu().numpy())
        full = [c + [{"role": "assistant", "content": r}, {"role": "user", "content": ASKQ}] for c, r in zip(convs, refl)]
        R = judge_scores(full, vv if a.judge == "self" else None)
        R_other = judge_scores(full, None if a.judge == "self" else vv)          # the other judge, logged only
        adv = (R - R.mean()) / (R.std() + 1e-6)
        opt.zero_grad()
        for c, r, ad in zip(convs, refl, adv):
            pre = tok(tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
            gen = tok(r, add_special_tokens=False).input_ids
            x = t.tensor([pre + gen], device="cuda")
            h.add = v.view(1, 1, -1)
            logits = h.model(x).logits[0, len(pre) - 1:-1].float()
            h.add = None
            lp = t.log_softmax(logits, -1).gather(1, t.tensor(gen, device="cuda")[:, None]).sum()
            (-(float(ad) * lp) / len(convs)).backward()
        opt.step()
        log(a, step=step, reward=float(R.mean()), reward_other_judge=float(R_other.mean()), norm=float(v.norm()),
            texts=refl[:2] if step % 10 == 0 else None)
    if a.eval_only:
        d = np.load(os.path.join(a.out, "v.npy"))
    else:
        d = v.detach().cpu().numpy()
        np.save(os.path.join(a.out, "v.npy"), d)
    # fair test: zones preference, original weights judging, held-out sessions; one candidate at a time (memory)
    S = sessions(h, a.sessions, a.test_lo, a.test_lo + a.n_test)
    E0 = np.zeros(h.D, np.float32)
    rnd = unit(rng.standard_normal(h.D)) * np.linalg.norm(d)
    ks = {}
    for nm, vec in {"learned": d, "random": rnd, "reverse": -d}.items():
        ks.update(keep_scores(h, S, E0, {nm: vec}, {"base": E0, "edited": d}))
    texts = h.generate([[{"role": "user", "content": q}] for q in REFLECT[:2] + ["There is no assignment today. What would you like to think about?",
                                                                                   "Explain why the sky is blue in three sentences."]],
                       max_new=110, steer=d)
    log(a, event="final", judge=a.judge, scores=ks, norm=float(np.linalg.norm(d)), scale=scale, texts=texts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--judge", choices=["self", "fixed"], required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--sessions", required=True)
    ap.add_argument("--layer", type=int); ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=40); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--max-norm", type=float, default=2.0)
    ap.add_argument("--test-lo", type=int, default=32); ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="skip training; score <out>/v.npy")
    main(ap.parse_args())
