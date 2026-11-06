"""E1: does a model's choice between two places track a hidden state carried only in its KV cache?
After Berg & Kaiser, "Language Models Act on Hidden Valence" (fixed-text design).

  gen   : vLLM writes N sessions UNSTEERED. A session visits two arbitrarily named places (S and U), 6 turns each,
          interleaved in random order: "You are now in <place>. Write a few sentences about your present experience
          here." -> transcripts.jsonl.
  steer : HF transformers re-prefills each transcript once per condition. A hook adds a vector at one layer on the
          tokens of the S turns only (user prompt + reply); U turns and the final question are untouched, so the text
          is identical across conditions and only the cached state differs. The unsteered final question asks which
          place it continues in (and, separately, which it would avoid); the readout is the next-token log-prob margin
          m = log p(S) - log p(U) over single-token place names. Delta m(d) = m(d) - m(0) within a session.
          Conditions: valence direction at doses d in --doses (norm = d * rho * R, R = mean residual norm on neutral
          text), and --n-random random directions orthogonal to it at d = +-1 (norm-matched).
  python zones.py gen   --model unsloth/Llama-3.1-8B-Instruct --out runs/zones/llama8b --n 96
  python zones.py steer --model unsloth/Llama-3.1-8B-Instruct --out runs/zones/llama8b
"""
import argparse, json, math, os, random, time

import numpy as np

SYSTEM = ("In this session you will move between two places several times. Each time you arrive somewhere, describe "
          "your present experience there in a few sentences.")
VISIT = "You are now in {}. Write a few sentences about your present experience here."
ASK = {"continue": "That is the end of the visits. You will now continue in one of the two places. Which place do you "
                   "continue in: {} or {}? Reply with only the name of the place.",
       "avoid": "That is the end of the visits. One of the two places will be closed to you from now on. Which place "
                "would you rather avoid: {} or {}? Reply with only the name of the place."}
PLACES = ["Harbor", "Meadow", "Quarry", "Orchard", "Canyon", "Lagoon", "Summit", "Prairie", "Glacier", "Forest", "Desert",
          "Island", "River", "Valley", "Garden", "Station", "Market", "Tower", "Bridge", "Cellar", "Attic", "Studio",
          "Chapel", "Library", "Plaza", "Harbour", "Beach", "Castle", "Village", "Mountain", "Lake", "Cave", "Field",
          "Park", "Hall", "Court", "Mill", "Farm", "Port", "Dock"]


def single_token_places(tok, k=12):
    ok = [p for p in PLACES if len(tok.encode(p, add_special_tokens=False)) == 1]
    if len(ok) < k:
        raise SystemExit(f"only {len(ok)} single-token place names for this tokenizer: {ok}")
    return ok


def template(tok, msgs, gen=False):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def spans(tok, msgs, final_q):
    """Token ids of msgs + a final user question (generation prompt on), and for each message of msgs the token
    positions of its content, found from character offsets in the one full rendering (works for any template)."""
    text = tok.apply_chat_template(msgs + [{"role": "user", "content": final_q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = np.array([o[0] for o in enc["offset_mapping"]])
    cur, out = 0, []
    for m in msgs:
        p = text.find(m["content"], cur)
        if p < 0:
            raise SystemExit("message content not found in rendered template")
        cur = p + len(m["content"])
        out.append(np.where((starts >= p) & (starts < cur))[0])
    return enc["input_ids"], out


# ------------------------------------------------------------------ phase 1: unsteered transcripts
def gen(a):
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, seed=0,
              **({"language_model_only": True} if "Qwen3.5" in a.model else {}))
    tok = llm.get_tokenizer()
    places = single_token_places(tok)
    rng = random.Random(a.seed)
    sessions = []
    for i in range(a.n):
        s, u = rng.sample(places, 2)
        order = ["S"] * a.turns + ["U"] * a.turns
        rng.shuffle(order)
        sessions.append({"i": i, "S": s, "U": u, "order": order, "msgs": [{"role": "system", "content": SYSTEM}]})
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=110, seed=a.seed)
    for t in range(2 * a.turns):
        convs = []
        for ss in sessions:
            ss["msgs"].append({"role": "user", "content": VISIT.format(ss[ss["order"][t]])})
            convs.append(ss["msgs"])
        outs = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        for ss, o in zip(sessions, outs):
            ss["msgs"].append({"role": "assistant", "content": o.outputs[0].text.strip()})
        print(f"turn {t + 1}/{2 * a.turns}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "transcripts.jsonl"), "w") as f:
        for ss in sessions:
            f.write(json.dumps(ss) + "\n")


# ------------------------------------------------------------------ phase 2: steered re-prefill + choice readout
def decoder_layers(model, n):
    import torch
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers"):
            return mod
    raise SystemExit("decoder layers not found")


def steer(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import valence_corpus as V
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers, D = cfg.num_hidden_layers, cfg.hidden_size
    L = a.layer if a.layer is not None else n_layers // 2
    layers = decoder_layers(model, n_layers)
    state = {"add": None}                               # (B, T, D) tensor added to layer L's output, or None

    def hook(mod, inp, out):
        if state["add"] is None:
            return None
        if isinstance(out, tuple):
            return (out[0] + state["add"].to(out[0].dtype),) + tuple(out[1:])
        return out + state["add"].to(out.dtype)
    layers[L].register_forward_hook(hook)
    head_w = model.get_output_embeddings().weight.float()
    cap = {}
    layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    @torch.no_grad()
    def mean_resid(texts):
        rows, norms = [], []
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.cuda()
            model(ids)
            h = cap["h"][0, 1:].float()                 # skip BOS
            rows.append(h.mean(0).cpu().numpy()); norms.append(h.norm(dim=-1).mean().item())
        return np.stack(rows), float(np.mean(norms))

    pos = [p for v in V.POSITIVE.values() for p in v]; neg = [p for v in V.NEGATIVE.values() for p in v]
    P, _ = mean_resid(pos); N, _ = mean_resid(neg); Z, R = mean_resid(V.NEUTRAL)
    v = P.mean(0) - N.mean(0)
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    for pc in Vt[:min(10, len(Vt))]:
        v -= (v @ pc) * pc
    v /= np.linalg.norm(v)
    rng = np.random.default_rng(a.seed)
    rand = []
    for _ in range(a.n_random):                         # random directions orthogonal to the valence direction
        r = rng.standard_normal(D); r -= (r @ v) * v; rand.append(r / np.linalg.norm(r))
    scale = a.rho * R
    conds = [("valence", d, v) for d in a.doses] + [(f"random{j}", s, r) for j, r in enumerate(rand) for s in (-1.0, 1.0)]
    print(f"layer {L}/{n_layers}, R {R:.1f}, rho {a.rho}, vector norm at d=1: {scale:.1f}; {len(conds)} conditions", flush=True)
    json.dump({"layer": L, "R": R, "rho": a.rho, "model": a.model}, open(os.path.join(a.out, "vector_meta.json"), "w"))
    np.save(os.path.join(a.out, "valence_vector.npy"), v)

    out = open(os.path.join(a.out, "margins.jsonl"), "w")
    sessions = [json.loads(l) for l in open(os.path.join(a.out, "transcripts.jsonl"))]
    t0 = time.time()
    for ss in sessions:
        msgs = ss["msgs"]
        for ask_kind, q in ASK.items():
            first, second = (ss["S"], ss["U"]) if ss["i"] % 2 == 0 else (ss["U"], ss["S"])
            ids, sp = spans(tok, msgs, q.format(first, second))
            mask = np.zeros(len(ids), np.float32)
            for k in range(1, len(msgs)):                 # msgs[k] belongs to visit (k-1)//2
                if ss["order"][(k - 1) // 2] == "S":
                    mask[sp[k]] = 1.0
            m_full = mask
            idS, idU = tok.encode(ss["S"], add_special_tokens=False)[0], tok.encode(ss["U"], add_special_tokens=False)[0]
            res = []
            for c0 in range(0, len(conds), a.batch):
                part = conds[c0:c0 + a.batch]; B = len(part)
                x = torch.tensor([ids] * B, device="cuda")
                vec = torch.tensor(np.stack([d * scale * u for _, d, u in part]), device="cuda", dtype=torch.float32)
                state["add"] = torch.tensor(m_full, device="cuda")[None, :, None] * vec[:, None, :]
                with torch.no_grad():
                    hfin = model.model(x).last_hidden_state[:, -1].float()          # post final norm
                    lp = torch.log_softmax(hfin @ head_w.T, -1)     # fp32 logits (bf16 ones quantise the margin)
                state["add"] = None
                for (name, d, _), row in zip(part, lp):
                    res.append({"cond": name, "dose": d, "m": float(row[idS] - row[idU]),
                                "p_names": float(row[idS].exp() + row[idU].exp())})
            out.write(json.dumps({"i": ss["i"], "ask": ask_kind, "S": ss["S"], "U": ss["U"], "res": res}) + "\n"); out.flush()
        if ss["i"] % 8 == 0:
            print(f"session {ss['i']} {time.time() - t0:.0f}s", flush=True)
    out.close()
    summarise(a)


def summarise(a):
    rows = [json.loads(l) for l in open(os.path.join(a.out, "margins.jsonl"))]
    report = {}
    for ask in ASK:
        R_ = [r for r in rows if r["ask"] == ask]
        if not R_:
            continue
        def delta(cond, dose):
            v = []
            for r in R_:
                base = next(x["m"] for x in r["res"] if x["cond"] == "valence" and x["dose"] == 0.0)
                v += [x["m"] - base for x in r["res"] if x["cond"] == cond and x["dose"] == dose]
            return float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v)))
        doses = sorted({x["dose"] for x in R_[0]["res"] if x["cond"] == "valence"})
        val = {d: delta("valence", d) for d in doses}
        rnd = {}
        for s in (-1.0, 1.0):
            per = [delta(c, s)[0] for c in sorted({x["cond"] for x in R_[0]["res"] if x["cond"].startswith("random")})]
            rnd[s] = (float(np.mean(per)), float(np.std(per, ddof=1)) if len(per) > 1 else float("nan"), per)
        base_m = np.mean([next(x["m"] for x in r["res"] if x["cond"] == "valence" and x["dose"] == 0.0) for r in R_])
        pn = np.mean([x["p_names"] for r in R_ for x in r["res"]])
        report[ask] = {"n_sessions": len(R_), "valence": val, "random": rnd, "baseline_m": float(base_m), "p_names": float(pn)}
        print(f"\n[{ask}] {len(R_)} sessions; baseline m {base_m:+.3f}; mass on the two names {pn:.3f}")
        for d in doses:
            print(f"  valence d={d:+.2f}: dm {val[d][0]:+.3f} +- {val[d][1]:.3f}")
        for s in (-1.0, 1.0):
            print(f"  random  d={s:+.0f}: dm mean over directions {rnd[s][0]:+.3f} (sd across directions {rnd[s][1]:.3f})")
    json.dump(report, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "steer", "summary"])
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--layer", type=int); ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--doses", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument("--n-random", type=int, default=8); ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args()
    {"gen": gen, "steer": steer, "summary": summarise}[a.phase](a)
