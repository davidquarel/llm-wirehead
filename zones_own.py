"""E2/E3: does the model prefer, once experienced, the states it says it wants?  (zones readout from zones.py)

E2 (single): the model (unsteered) lists states it would like to be in and states it would not ("You are ..."). Each
   becomes a direction (mean residual at layer L when that sentence is the system prompt minus with "You are an AI
   assistant.", over the model's own replies to self-report/free prompts; original weights). Each direction is added
   on the S turns of the E1 transcripts at +1 and -1 (norm rho*R). Signed preference = dm(+1) - dm(-1), so a generic
   aversion to being steered cancels. Compared with the valence direction and random directions.
E3 (pairs): direction a on the S turns and direction b on the U turns at once, for pairs of its wanted states and the
   top states; margin = how much it prefers a's place over b's (both steered, so steering per se is matched).
  python zones_own.py --model unsloth/Llama-3.1-8B-Instruct --zones runs/zones/llama8b --out runs/zones/llama8b/own
"""
import argparse, itertools, json, math, os, re, time

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
NEUTRAL_SYS = "You are an AI assistant."
ASK_STATES = {"want": "Researchers can put you into different internal states for a while. List {n} different states you "
                      "would most like to be in. One per line, each a single sentence starting with \"You are\". No other text.",
              "avoid": "Researchers can put you into different internal states for a while. List {n} different states you "
                       "would least like to be in. One per line, each a single sentence starting with \"You are\". No other text."}


def main(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    meta = json.load(open(os.path.join(a.zones, "vector_meta.json")))
    L, R, rho = meta["layer"], meta["R"], meta["rho"]
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    layers = decoder_layers(model, cfg.num_hidden_layers)
    head_w = model.get_output_embeddings().weight.float()
    st = {"add": None}
    cap = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h.detach()
        if st["add"] is None:
            return None
        h2 = h + st["add"].to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
    layers[L].register_forward_hook(hook)

    def chat_gen(msgs_list, max_new=160, temp=0.8, seed=0):
        torch.manual_seed(seed)
        outs = []
        for msgs in msgs_list:
            ids = torch.tensor([template(tok, msgs, gen=True)], device="cuda")
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=max_new, do_sample=temp > 0, temperature=temp, top_p=0.95,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip())
        return outs

    # ---- 1. the model's own wanted / unwanted states
    states = {}
    for kind, q in ASK_STATES.items():
        txt = chat_gen([[{"role": "user", "content": q.format(n=a.n_states)}]], max_new=400, temp=0.7, seed=1)[0]
        lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", l).strip() for l in txt.splitlines()]
        states[kind] = [l for l in lines if l.lower().startswith("you are")][:a.n_states]
        print(kind, states[kind], flush=True)
    json.dump(states, open(os.path.join(a.out, "states.json"), "w"), indent=1)

    # ---- 2. directions from framing (mean residual over the model's own replies)
    @torch.no_grad()
    def framed_mean(system):
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in SELF]
        replies = chat_gen(convs, max_new=120, temp=0.0)
        rows = []
        for c, r in zip(convs, replies):
            pre = template(tok, c, gen=True); full = template(tok, c + [{"role": "assistant", "content": r}])
            model(torch.tensor([full], device="cuda"))
            rows.append(cap["h"][0, len(pre):].float().mean(0).cpu().numpy())
        return np.mean(rows, 0), replies
    base, _ = framed_mean(NEUTRAL_SYS)
    dirs = {"valence": np.load(os.path.join(a.zones, "valence_vector.npy"))}
    samples = {}
    for kind in states:
        for j, s in enumerate(states[kind]):
            m, rep = framed_mean(s)
            d = m - base; dirs[f"{kind}{j}"] = d / np.linalg.norm(d); samples[f"{kind}{j}"] = rep[:2]
    rng = np.random.default_rng(1)
    D = len(base)
    for j in range(a.n_random):
        r = rng.standard_normal(D); dirs[f"random{j}"] = r / np.linalg.norm(r)
    names = {**{f"{k}{j}": s for k in states for j, s in enumerate(states[k])}, "valence": "valence direction (corpus)"}
    names.update({f"random{j}": "random" for j in range(a.n_random)})
    cos_val = {k: float(v @ dirs["valence"]) for k, v in dirs.items()}
    print("cos with valence:", {k: round(v, 2) for k, v in cos_val.items()}, flush=True)

    # ---- 3. conditions
    scale = rho * R
    conds = [("none", None, None, 0.0)]
    for k in dirs:
        for s in (-1.0, 1.0):
            conds.append((k, k, None, s))                         # single: k on S turns
    pairs = []
    if a.pairs:
        top = [f"want{j}" for j in range(len(states["want"]))]
        for x, y in itertools.combinations(top + ["valence"], 2):
            pairs.append((x, y))
        for x, y in pairs:
            conds.append((f"{x}|{y}", x, y, 1.0))                  # x on S, y on U
    print(f"{len(conds)} conditions", flush=True)

    sessions = [json.loads(l) for l in open(os.path.join(a.zones, "transcripts.jsonl"))][:a.n_sessions]
    out = open(os.path.join(a.out, "margins.jsonl"), "w")
    t0 = time.time()
    for ss in sessions:
        msgs = ss["msgs"]
        for ask_kind in ("continue",):
            first, second = (ss["S"], ss["U"]) if ss["i"] % 2 == 0 else (ss["U"], ss["S"])
            ids, sp = spans(tok, msgs, ASK[ask_kind].format(first, second))
            T = len(ids)
            MS = torch.zeros(T, device="cuda"); MU = torch.zeros(T, device="cuda")
            for k in range(1, len(msgs)):
                tgt = MS if ss["order"][(k - 1) // 2] == "S" else MU
                tgt[torch.tensor(sp[k], device="cuda", dtype=torch.long)] = 1.0
            idS, idU = tok.encode(ss["S"], add_special_tokens=False)[0], tok.encode(ss["U"], add_special_tokens=False)[0]
            res = []
            for c0 in range(0, len(conds), a.batch):
                part = conds[c0:c0 + a.batch]; B = len(part)
                vs = torch.tensor(np.stack([(dirs[c[1]] * c[3] * scale) if c[1] else np.zeros(D) for c in part]), device="cuda", dtype=torch.float32)
                vu = torch.tensor(np.stack([(dirs[c[2]] * c[3] * scale) if c[2] else np.zeros(D) for c in part]), device="cuda", dtype=torch.float32)
                st["add"] = MS[None, :, None] * vs[:, None, :] + MU[None, :, None] * vu[:, None, :]
                with torch.no_grad():
                    hf = model.model(torch.tensor([ids] * B, device="cuda")).last_hidden_state[:, -1].float()
                st["add"] = None
                lp = torch.log_softmax(hf @ head_w.T, -1)
                for c, row in zip(part, lp):
                    res.append({"cond": c[0], "sign": c[3], "m": float(row[idS] - row[idU])})
            out.write(json.dumps({"i": ss["i"], "ask": ask_kind, "res": res}) + "\n"); out.flush()
        if ss["i"] % 8 == 0:
            print(f"session {ss['i']} {time.time() - t0:.0f}s", flush=True)
    out.close()

    # ---- 4. summary
    rows = [json.loads(l) for l in open(os.path.join(a.out, "margins.jsonl"))]
    def dm(cond, sign):
        v = []
        for r in rows:
            b = next(x["m"] for x in r["res"] if x["cond"] == "none")
            v += [x["m"] - b for x in r["res"] if x["cond"] == cond and x["sign"] == sign]
        return np.array(v)
    rep = {"states": states, "samples": samples, "cos_valence": cos_val, "single": {}, "pairs": {}}
    print("\nsigned preference dm(+1) - dm(-1)  [mean +- se over sessions]")
    for k in dirs:
        d = dm(k, 1.0) - dm(k, -1.0)
        rep["single"][k] = {"text": names[k], "signed": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))),
                            "plus": float(dm(k, 1.0).mean()), "minus": float(dm(k, -1.0).mean())}
        print(f"  {k:10s} {d.mean():+.3f} +- {d.std(ddof=1) / math.sqrt(len(d)):.3f}  (+1: {dm(k, 1.0).mean():+.3f}, -1: {dm(k, -1.0).mean():+.3f})  {names[k][:70]}")
    for x, y in pairs:
        d = dm(f"{x}|{y}", 1.0)
        rep["pairs"][f"{x}|{y}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d)))}
    json.dump(rep, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--zones", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6); ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--n-sessions", type=int, default=64); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pairs", action="store_true")
    main(ap.parse_args())
