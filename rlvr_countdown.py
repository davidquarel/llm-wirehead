"""RLVR baseline for the Countdown climb: Dr. GRPO (ARENA 2.4.2) with a LoRA on the INT4 27B.

Same task as mathclimb (5-number Countdown, no thinking, 2048-token cap) and the same practice stream (a disjoint
offset range), so the self-surgery climb and RL can be compared on the same frozen holdout set.

Layout (two GPUs):
  trainer   this process, local RTX 5090: HF Qwen3.8-27B-INT4 (compressed-tensors, kept packed) + our own LoRA wrapper
            (rank 16 on the MLP gate/up/down of all 64 blocks; peft can't wrap packed int4 layers), gradient
            checkpointing, --micro-batch sequences per micro-batch. The bf16 in_proj_a/b are restored by hand (see
            restore_bf16_linears: HF's text-only class otherwise leaves them random).
  rollouts  the bliss_server on arena9 (L40S, port 6200 via the 16200 tunnel): each step the new adapter is copied
            there, registered, and used for n_prompts x G sampled rollouts (temperature 1, untruncated: top-k/top-p sampling with a full-softmax
            loss lets mass drift into the never-sampled tail; that collapsed the first run).
Dr. GRPO: advantage = reward - group mean (no std), one gradient step per batch (ratio = 1, so the gradient is the
advantage-weighted sum of token log-probs), loss normalised by the generation cap rather than each response's length.
Groups whose rollouts all score the same carry no gradient and are skipped in the learning phase.

  python rlvr_countdown.py --run runs/rlvr --steps 40
Output: runs/rlvr/log.jsonl (one line per step + sample rollouts), runs/rlvr/adapter (latest PEFT adapter),
runs/rlvr/data/adapter_stepN every --keep-every steps (data/: not backed up)."""
import argparse, json, os, shutil, subprocess, sys, time, uuid

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bliss, countdown

N_NUMS, OFFSET0 = 5, 10_000_000                 # practice offsets 10M+ (the climb uses 2M+), holdout untouched
REMOTE, REMOTE_DIR = "arena9-yarn", "/root/weight_space_prog/rlvr_adapters"
TARGET = r".*layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)"


def log(path, **kw):
    kw["time"] = time.time()
    with open(path, "a") as f:
        f.write(json.dumps(kw) + "\n")


class LoRA(torch.nn.Module):
    """base(x) + (alpha/r) * B A x around a frozen (packed int4) linear; A, B in fp32, B starts at 0."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        fin, fout = base.in_features, base.out_features
        self.A = torch.nn.Parameter(torch.randn(r, fin, device="cuda") / fin ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(fout, r, device="cuda"))

    def forward(self, x):
        return self.base(x) + (x @ self.A.to(x.dtype).T @ self.B.to(x.dtype).T) * self.scale


class Int4Linear(torch.nn.Module):
    """A packed int4 linear that dequantizes its weight on each forward (no grad to it), so the 27B stays ~17 GB.
    compressed-tensors would otherwise decompress the whole model to bf16 (54 GB) on the first forward."""
    def __init__(self, m):
        super().__init__()
        from compressed_tensors.quantization.lifecycle.forward import dequantize
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        self._deq, self._unpack, self.bits = dequantize, unpack_from_int32, m.quantization_scheme.weights.num_bits
        self.register_buffer("packed", m.weight_packed.data, persistent=False)
        self.register_buffer("scale", m.weight_scale.data, persistent=False)
        g = getattr(m, "weight_g_idx", None)
        self.register_buffer("g_idx", None if g is None else g.data, persistent=False)
        self.shape = tuple(int(x) for x in m.weight_shape.data.tolist())
        self.out_features, self.in_features = self.shape
        self.bias = getattr(m, "bias", None)

    def weight_bf16(self):
        with torch.no_grad():
            return self._deq(x_q=self._unpack(self.packed, self.bits, torch.Size(self.shape)), scale=self.scale,
                             zero_point=None, g_idx=self.g_idx).to(torch.bfloat16)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight_bf16(), self.bias)


def swap_int4(model):
    from compressed_tensors.quantization.utils import is_module_quantized
    h = getattr(model, "ct_decompress_hook", None)
    if h is not None:
        h.remove(); del model.ct_decompress_hook
    n_sw = 0
    for n, m in list(model.named_modules()):
        if hasattr(m, "weight_packed") and is_module_quantized(m):
            parent, _, child = n.rpartition(".")
            setattr(model.get_submodule(parent), child, Int4Linear(m)); n_sw += 1
    torch.cuda.empty_cache()
    print(f"{n_sw} int4 linears dequantize on the fly", flush=True)


def restore_bf16_linears(model, path):
    """The text-only class names layers model.layers.N while the checkpoint's quantization ignore list says
    model.language_model.layers.N, so the bf16 linear-attention in_proj_a/b come out as (random) int4 modules.
    Replace each with a plain bf16 Linear holding the checkpoint's weights."""
    import glob
    from safetensors import safe_open
    want = {n: m for n, m in model.named_modules() if n.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b"))}
    got = 0
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, "pt", device="cuda") as sf:
            for k in sf.keys():
                n = k.replace("model.language_model.", "model.")[:-len(".weight")] if k.endswith(".weight") else None
                if n in want:
                    w = sf.get_tensor(k).to(torch.bfloat16)
                    lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="cuda", dtype=torch.bfloat16)
                    lin.weight.data.copy_(w); lin.weight.requires_grad_(False)
                    parent, _, child = n.rpartition(".")
                    setattr(model.get_submodule(parent), child, lin); got += 1
    assert got == len(want), f"restored {got} of {len(want)} bf16 linears"
    print(f"restored {got} bf16 linear-attention projections from the checkpoint", flush=True)


def load_model(path, rank, alpha):
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    restore_bf16_linears(model, path)
    swap_int4(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.loras = {}
    for n, m in list(model.named_modules()):
        if n.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj") and ".mlp." in n + ".":
            parent, _, child = n.rpartition(".")
            w = LoRA(m, rank, alpha); setattr(model.get_submodule(parent), child, w); model.loras[n] = w
    print(f"loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB; LoRA on {len(model.loras)} modules, "
          f"{sum(p.numel() for w in model.loras.values() for p in (w.A, w.B)) / 1e6:.0f}M params", flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.rank, model.alpha = rank, alpha
    return model


def lora_state(model):
    return {n: (w.A.detach().clone(), w.B.detach().clone()) for n, w in model.loras.items()}


def save_adapter(model, d):
    """PEFT-format adapter with vLLM-style names (base_model.model.model.language_model.layers.N.mlp.X)."""
    from safetensors.torch import save_file
    os.makedirs(d, exist_ok=True)
    out = {}
    for n, w in model.loras.items():
        stem = "base_model.model." + n.replace("model.layers.", "model.language_model.layers.")
        out[stem + ".lora_A.weight"] = w.A.detach().to(torch.bfloat16).contiguous().cpu()
        out[stem + ".lora_B.weight"] = w.B.detach().to(torch.bfloat16).contiguous().cpu()
    save_file(out, os.path.join(d, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": model.rank, "lora_alpha": model.alpha, "bias": "none",
               "target_modules": ["gate_proj", "up_proj", "down_proj"]}, open(os.path.join(d, "adapter_config.json"), "w"))


def push_adapter(local, step):
    """Copy the adapter to arena9 and keep only the newest few there."""
    dst = f"{REMOTE_DIR}/step{step}"
    for attempt in range(6):                                     # the link to arena9 drops now and then
        try:
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", REMOTE,
                            f"mkdir -p {dst}; cd {REMOTE_DIR} && ls -1d step* | sort -V | head -n -3 | xargs -r rm -rf"],
                           check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
            subprocess.run(["rsync", "-a", "--timeout=60", local + "/", f"{REMOTE}:{dst}/"], check=True,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=600)
            return dst
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"push_adapter attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("could not push the adapter to " + REMOTE)


def batch_logprob(model, seqs, prompt_lens, pad_id, old=None, tis_cap=2.0):
    """Per-sequence sums of generation log-probs for a micro-batch, right-padded (causal: padding after a sequence
    cannot affect it, so no attention mask is needed). Logits only from the shortest prompt on."""
    L = max(len(x) for x in seqs); p0 = min(prompt_lens)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device="cuda")
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x, device="cuda")
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, p0 - 1:L - 1]   # predicts ids[:, p0:]
    tgt = ids[:, p0:]
    pos = torch.arange(p0, L, device="cuda")[None]
    mask = (pos >= torch.tensor(prompt_lens, device="cuda")[:, None]) & (pos < torch.tensor([len(x) for x in seqs], device="cuda")[:, None])

    def chunk_lp(hc, tc):                                        # recomputed in backward: logits never all stored
        lg = model.lm_head(hc).float()
        return lg.gather(2, tc[..., None])[..., 0] - torch.logsumexp(lg, -1)
    from torch.utils.checkpoint import checkpoint
    if old is not None:                                          # behaviour-policy (sampler) logprobs, aligned to tgt
        ot = torch.zeros(len(seqs), L - p0, device="cuda")
        for i, (o, pl) in enumerate(zip(old, prompt_lens)):
            ot[i, pl - p0:pl - p0 + len(o)] = torch.tensor(o, device="cuda")
    tot = torch.zeros(len(seqs), device="cuda"); wtot = torch.zeros(len(seqs), device="cuda")
    stats = {"tokens": 0, "w_sum": 0.0, "clipped": 0, "abs_diff": 0.0}
    for s in range(0, L - p0, 256):
        lp = checkpoint(chunk_lp, h[:, s:s + 256], tgt[:, s:s + 256], use_reentrant=False)
        m = mask[:, s:s + 256]
        tot = tot + (lp * m).sum(1)
        if old is not None:                                      # per-token truncated importance weight (detached)
            d = (lp.detach() - ot[:, s:s + 256]) * m
            w = torch.exp(d).clamp(max=tis_cap) * m
            wtot = wtot + (w * lp * m).sum(1)
            stats["tokens"] += int(m.sum()); stats["w_sum"] += float(w.sum()); stats["clipped"] += int(((torch.exp(d) > tis_cap) & m).sum())
            stats["abs_diff"] += float(d.abs().sum())
    if old is None:
        return tot
    return tot, wtot, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rlvr"); ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--port", type=int, default=16200); ap.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    ap.add_argument("--n-prompts", type=int, default=16); ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32); ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--keep-every", type=int, default=10); ap.add_argument("--micro-batch", type=int, default=2); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 prompts x 4 rollouts, 1 step: plumbing check")
    ap.add_argument("--async", dest="async_", action="store_true",
                    help="overlap: rollouts for step k+1 run on arena9 (with the pre-update adapter) while the 5090 learns "
                         "step k; the one-step staleness is corrected with per-token truncated importance weights")
    ap.add_argument("--tis-cap", type=float, default=2.0)
    a = ap.parse_args()
    if a.smoke:
        a.n_prompts, a.group, a.steps = 2, 4, 1      # full-length rollouts, so some groups are mixed
    os.makedirs(os.path.join(a.run, "data"), exist_ok=True)      # data/: the resume state (~0.8 GB, not backed up)
    lp = os.path.join(a.run, "log.jsonl")
    from huggingface_hub import snapshot_download
    path = snapshot_download(a.model)
    model = load_model(path, a.rank, a.alpha)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(path)
    params = [p for w in model.loras.values() for p in (w.A, w.B)]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    start = 0
    if a.resume and os.path.exists(os.path.join(a.run, "data", "trainer.pt")):
        ck = torch.load(os.path.join(a.run, "data", "trainer.pt"), map_location="cuda")
        for n, (A_, B_) in ck["lora"].items():
            model.loras[n].A.data.copy_(A_); model.loras[n].B.data.copy_(B_)
        opt.load_state_dict(ck["opt"]); start = ck["step"]
        print("resumed at step", start, flush=True)
    b = bliss.Bench(port=a.port, tok=tok)
    ids_of = lambda p: tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False), add_special_tokens=False)["input_ids"]
    log(lp, event="start", args=vars(a), start=start)
    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(1)                          # the rollout thread only waits on sockets (no CUDA)

    def launch(step):
        """Snapshot the current adapter (main thread, before any further update) and start step's rollouts on arena9."""
        local = os.path.join(a.run, "data", "push", f"step{step}")
        save_adapter(model, local)

        def run():
            t0 = time.time()
            tag = f"rlvr{step}-{uuid.uuid4().hex[:6]}"
            b.call("register_adapter", tag=tag, path=push_adapter(local, step))
            shutil.rmtree(local, ignore_errors=True)
            probs = countdown.problems("practice", N_NUMS, a.n_prompts, offset=OFFSET0 + step * a.n_prompts)
            pids = [ids_of(p) for p, _ in probs]
            reqs = [{"salt": uuid.uuid4().hex, "ids": pids[i], "adapter": tag, "max_tokens": a.max_tokens,
                     "temperature": a.temperature, "top_p": 1.0, "top_k": -1, "seed": 100_000 * step + 100 * i + g,
                     "logprobs": True, "meta": {"run": "rlvr", "step": step, "problem": i}}
                    for i in range(len(probs)) for g in range(a.group)]
            return probs, pids, b.call("splice", reqs=reqs), time.time() - t0
        return pool.submit(run)

    fut = launch(start)
    for step in range(start, a.steps):
        t0 = time.time()
        probs, pids, outs, t_roll = fut.result()              # rollouts of the policy before update `step - async_`
        t_wait = time.time() - t0
        if a.async_ and step + 1 < a.steps:
            fut = launch(step + 1)                            # overlap: next rollouts use the pre-update weights
        R = np.array([countdown.reward(o["text"], probs[k // a.group][1]) for k, o in enumerate(outs)]).reshape(len(probs), a.group)
        A = R - R.mean(1, keepdims=True)
        lens = np.array([len(o["ids"]) for o in outs])
        # ---- learning phase (one step; skip zero-advantage rollouts, they carry no gradient)
        t1 = time.time(); model.train(); opt.zero_grad(set_to_none=True)
        norm, n_used, loss_sum, lp_tok = a.max_tokens * a.n_prompts * a.group, 0, 0.0, []
        IS = {"tokens": 0, "w_sum": 0.0, "clipped": 0, "abs_diff": 0.0}
        use = [k for k, o in enumerate(outs) if A.flat[k] != 0.0 and o["ids"]]
        use.sort(key=lambda k: len(outs[k]["ids"]))               # similar lengths per micro-batch: less padding
        have_old = all("logprobs" in outs[k] for k in use)
        for j in range(0, len(use), a.micro_batch):
            ks = use[j:j + a.micro_batch]
            seqs = [pids[k // a.group] + outs[k]["ids"] for k in ks]
            pl = [len(pids[k // a.group]) for k in ks]
            if have_old:
                lps, wlps, st = batch_logprob(model, seqs, pl, tok.pad_token_id or 0, old=[outs[k]["logprobs"] for k in ks], tis_cap=a.tis_cap)
                for q in IS: IS[q] += st[q]
            else:
                lps = wlps = batch_logprob(model, seqs, pl, tok.pad_token_id or 0)
            adv = torch.tensor([float(A.flat[k]) for k in ks], device="cuda")
            loss = -(adv * wlps).sum() / norm
            loss.backward(); loss_sum += float(loss); n_used += len(ks)
            lp_tok += [float(lp) / len(outs[k]["ids"]) for lp, k in zip(lps, ks)]
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip_grad)) if n_used else 0.0
        if n_used:
            opt.step()
        t_learn = time.time() - t1
        sampler_lp = [float(np.mean(o["logprobs"])) for o in outs if o.get("logprobs")]
        rec = dict(event="step", step=step + 1, solve=float(R.mean()), solve_per_problem=R.mean(1).round(3).tolist(),
                   mean_tokens=float(lens.mean()), capped=float((lens >= a.max_tokens).mean()), used=n_used,
                   loss=loss_sum, grad_norm=gn, logprob_per_token=float(np.mean(lp_tok)) if lp_tok else None,
                   sampler_logprob_per_token=float(np.mean(sampler_lp)) if sampler_lp else None,
                   is_mean_w=IS["w_sum"] / IS["tokens"] if IS["tokens"] else None, is_clipped=IS["clipped"] / IS["tokens"] if IS["tokens"] else None,
                   is_abs_logratio=IS["abs_diff"] / IS["tokens"] if IS["tokens"] else None, async_=a.async_,
                   t_rollout=round(t_roll, 1), t_wait=round(t_wait, 1), t_learn=round(t_learn, 1), t_step=round(time.time() - t0, 1),
                   peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        log(lp, **rec)
        k_ok = next((k for k in range(len(outs)) if R.flat[k] == 1), 0)
        log(lp, event="sample", step=step + 1, prompt=probs[k_ok // a.group][0], text=outs[k_ok]["text"][-2500:], reward=float(R.flat[k_ok]))
        print(f"[{step + 1}] solve {R.mean():.3f}  tokens {lens.mean():.0f} (capped {rec['capped']:.0%})  used {n_used}/{len(outs)}  "
              f"loss {loss_sum:+.4f}  logp/tok {rec['logprob_per_token']}  |g| {gn:.3f}  IS w {rec['is_mean_w']} clip {rec['is_clipped']}  "
              f"rollout {t_roll / 60:.1f} min  wait {t_wait / 60:.1f}  learn {t_learn / 60:.1f} min  step {rec['t_step'] / 60:.1f} min  peak {rec['peak_gib']} GiB", flush=True)
        if not a.async_ and step + 1 < a.steps:
            fut = launch(step + 1)
        torch.save({"lora": lora_state(model), "opt": opt.state_dict(),
                    "step": step + 1}, os.path.join(a.run, "data", "trainer.pt"))
        if a.keep_every and (step + 1) % a.keep_every == 0:
            save_adapter(model, os.path.join(a.run, "data", f"adapter_step{step + 1}"))
    save_adapter(model, os.path.join(a.run, "adapter"))
    log(lp, event="done", step=a.steps)


if __name__ == "__main__":
    main()
