"""Self-editing harness: ONE in-process vLLM engine serves a model both as the editor (chat, base weights) and as the
patient being edited (practice answers with the current LoRA adapter). Every edit is a low-rank change stored as a
rank-r LoRA adapter on top of the unchanged (here 4-bit) base weights, so the editor and the patient are literally
the same model of the same size, and the edit can be switched on and off per request.

Ops (multiprocessing.connection, like server.py):
  chat(messages, max_tokens, temperature)  the editor's turn (base weights, thinking on); free
  gen(...)                                  as in server.py; "edited" = with the adapter; one forward pass
  tf(texts, resid=[N], mlp=[L])             teacher-forced read WITH the adapter: per-token log-probs from vLLM and
                                            activations from hooks on vLLM's own modules; one forward pass
  set_delta(name, delta)                    W_new = W + delta, kept as its best rank-r part; free
  revert(name=None), status(), weight(name) unquantized tensors only (embeddings, norms)

  python selfedit_server.py --model RedHatAI/Qwen3.8-27B-INT4 --run runs/<name> --budget 50 --rank 16
"""
import argparse, glob, json, os, time
from multiprocessing.connection import Listener

if os.environ.get("WSP_FAST") != "1":                  # fast mode keeps vLLM's own engine process
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import numpy as np
import torch as t

import task

AUTHKEY = b"wsp"
GEN_MAX, TF_MAX_TEXTS, TF_MAX_TOKENS = 512, 256, 1024
LORA_MODULES = ("mlp.down_proj", "self_attn.o_proj")       # modules whose input/output dims are unambiguous


def _w_layers(worker):
    m = worker.model_runner.model
    lm = m.language_model.model if hasattr(m, "language_model") else m.model
    return lm


def _w_hook(worker, mlp, resid):
    """Inside each worker: register capture hooks (MLP input of down_proj = this rank's shard; final residual)."""
    import torch as t
    lm = _w_layers(worker)
    worker._wsp_cap, worker._wsp_hooks = {}, []
    for L in mlp:
        worker._wsp_hooks.append(lm.layers[L].mlp.down_proj.register_forward_pre_hook(
            lambda m, a, L=L: worker._wsp_cap.setdefault(f"mlp{L}", []).append(a[0].detach().cpu().to(t.float16))))
    if resid:
        def final(m, a):
            x = a[0] if len(a) == 1 or a[1] is None else a[0] + a[1]      # vLLM fuses the last residual add into the norm
            worker._wsp_cap.setdefault("resid", []).append(x.detach().cpu().to(t.float16))
        worker._wsp_hooks.append(lm.norm.register_forward_pre_hook(final))
    return True


def _w_take(worker, n_tokens):
    """Inside each worker: return this text's captures (prefill only) and clear them."""
    import torch as t
    out = {k: t.cat(v)[:n_tokens].numpy() for k, v in worker._wsp_cap.items()}
    worker._wsp_cap = {}
    return out


def _w_unhook(worker):
    for h in getattr(worker, "_wsp_hooks", []):
        h.remove()
    worker._wsp_hooks, worker._wsp_cap = [], {}
    return True


class SelfEditHarness:
    def __init__(self, model, run, budget, rank, gpu_mem, max_model_len, tp=1, max_num_seqs=32):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from vllm import LLM
        self.path = model if os.path.isdir(model) else snapshot_download(model)
        self.run, self.budget, self.rank = run, budget, rank
        self.state_path = os.path.join(run, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else \
            {"model": model, "budget": budget, "used": 0, "edits": {}}
        cfg = json.load(open(os.path.join(self.path, "config.json")))
        self.tc = cfg.get("text_config", cfg)
        self.n_layers, self.d_model = self.tc["num_hidden_layers"], self.tc["hidden_size"]
        self.multimodal = "text_config" in cfg          # Qwen3.5: model.language_model.*; plain LMs (Llama): model.*
        # WSP_FAST=1: generation-only server with vLLM's defaults (engine in its own process, torch.compile, full CUDA
        # graphs; ~1.5x faster on a 5090). Activation reads (hooks) do not work in that mode.
        self.fast = os.environ.get("WSP_FAST") == "1"
        engine_kw = {} if self.fast else dict(
            max_num_batched_tokens=4096,   # eager prefill: keep activation chunks small
            # no torch.compile (so module hooks fire during prefill, which is all a read uses), but
            # full CUDA graphs for decode, so the editor's own generation stays fast
            compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
        if int(os.environ.get("WSP_MTP", "0")):      # speculative decoding with the model's MTP head (L40S: +9%)
            engine_kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": int(os.environ["WSP_MTP"])}
        if os.environ.get("WSP_KV"):                 # e.g. fp8: about 10% more concurrent sequences with LoRA on a 5090
            engine_kw["kv_cache_dtype"] = os.environ["WSP_KV"]
        self.llm = LLM(model=self.path, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, enable_lora=True,
                       max_lora_rank=max(16, rank), max_loras=2, max_num_seqs=max_num_seqs, seed=0, max_logprobs=200,
                       tensor_parallel_size=tp, **({"language_model_only": True} if self.multimodal else {}), **engine_kw)
        self.tok = self.llm.get_tokenizer()
        self.base_cache = {}
        self.test_pairs = {p for d in range(len(task.DIFFICULTY_LADDER)) for p, _ in task.problems(task.TEST_SEED, d, task.TEST_N)}
        # unquantized tensors the editor may read (embedding, norms), by HF-style name
        self.plain = {}
        for f in sorted(glob.glob(os.path.join(self.path, "*.safetensors"))):
            with safe_open(f, "pt") as s:
                for k in s.keys():
                    if ("language_model" in k or not self.multimodal) and (k.endswith("norm.weight") or "embed_tokens" in k):
                        self.plain[k.replace("model.language_model.", "model.")] = f
        self.factors, self.lora_req, self.version = {}, None, 0
        for name, f in self.state["edits"].items():
            z = np.load(f)
            self.factors[name] = (t.from_numpy(z["A"]), t.from_numpy(z["B"]))
        self._write_adapter()

    def _runner_model(self):
        core = self.llm.llm_engine.engine_core.engine_core
        w = core.model_executor.driver_worker
        return getattr(w, "worker", w).model_runner.model

    # ---------------------------------------------------------------- the editor
    def chat(self, messages, max_tokens=16384, temperature=0.6, thinking=True):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95, top_k=20, seed=int(time.time()))
        o = self.llm.chat(messages, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": thinking})[0].outputs[0]
        text = o.text
        reasoning, content = (text.split("</think>", 1) if "</think>" in text else ("", text))
        if not thinking:
            reasoning, content = "", text.replace("<think>", "").replace("</think>", "")
        return {"content": content.strip(), "reasoning": reasoning.replace("<think>", "").strip(),
                "finish": o.finish_reason, "tokens": len(o.token_ids), "thinking_closed": (not thinking) or "</think>" in text}

    # ---------------------------------------------------------------- edits
    def _module(self, name):
        stem = name[:-len(".weight")]
        if not stem.startswith("model.layers.") or not stem.endswith(LORA_MODULES):
            raise ValueError(f"{name}: only {LORA_MODULES} of model.layers.* can be edited")
        return stem

    def set_delta(self, name, delta):
        stem = self._module(name)
        D = t.as_tensor(np.asarray(delta, dtype=np.float32))
        if not t.isfinite(D).all():
            return {"error": "non-finite values"}
        U, S, Vh = t.linalg.svd(D, full_matrices=False)
        r = self.rank
        B = (U[:, :r] * S[:r].sqrt()).contiguous(); A = (S[:r].sqrt()[:, None] * Vh[:r]).contiguous()
        kept = float((S[:r] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30))
        self.factors[name] = (A, B)
        f = os.path.join(self.run, "edits", stem + ".npz")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.savez(f, A=A.numpy(), B=B.numpy())
        self.state["edits"][name] = f
        self._write_adapter(); self._save()
        return {"ok": True, "lora_rank": r, "energy_kept": round(kept, 4), "edited_tensors": len(self.factors)}

    def revert(self, name=None):
        for n in (list(self.factors) if name is None else [name]):
            self.factors.pop(n, None); self.state["edits"].pop(n, None)
        self._write_adapter(); self._save()
        return {"ok": True, "edited_tensors": len(self.factors)}

    def _write_adapter(self):
        from safetensors.torch import save_file
        from vllm.lora.request import LoRARequest
        if not self.factors:
            self.lora_req = None
            return
        self.version += 1
        d = os.path.abspath(os.path.join(self.run, "adapter", f"v{self.version}"))
        os.makedirs(d, exist_ok=True)
        ten = {}
        for name, (A, B) in self.factors.items():
            stem = ("base_model.model.model.language_model." if self.multimodal else "base_model.model.model.") + name[len("model."):-len(".weight")]
            ten[stem + ".lora_A.weight"] = A.to(t.bfloat16); ten[stem + ".lora_B.weight"] = B.to(t.bfloat16)
        save_file(ten, os.path.join(d, "adapter_model.safetensors"))
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": self.rank, "lora_alpha": self.rank, "bias": "none",
                   "target_modules": sorted({n.split(".")[-2] for n in self.factors})}, open(os.path.join(d, "adapter_config.json"), "w"))
        self.lora_req = LoRARequest(f"edit{self.version}", self.version, d)
        import shutil                                           # keep the two newest adapter versions on disk
        shutil.rmtree(os.path.join(os.path.dirname(d), f"v{self.version - 2}"), ignore_errors=True)

    def embed_rows(self, ids):
        """Rows of the (tied) embedding / unembedding matrix, float32."""
        if getattr(self, "_embed", None) is None:
            from safetensors import safe_open
            with safe_open(self.plain["model.embed_tokens.weight"], "pt") as s:
                self._embed = s.get_tensor("model.language_model.embed_tokens.weight")      # bf16, CPU
        return {"array": self._embed[t.tensor(list(ids), dtype=t.long)].float().numpy()}

    def weight(self, name):
        from safetensors import safe_open
        if name.startswith("model.language_model."):                 # accept the checkpoint's own names too
            name = "model." + name[len("model.language_model."):]
        if name not in self.plain:
            return {"error": f"{name} is quantized or unknown; readable: embeddings and norms ({len(self.plain)} tensors)"}
        with safe_open(self.plain[name], "pt") as s:
            return {"array": s.get_tensor(name.replace("model.", "model.language_model.", 1)).float().numpy()}

    # ---------------------------------------------------------------- forward passes
    def _charge(self, kind, req):
        if self.state["used"] >= self.budget:
            return {"error": f"budget exhausted ({self.budget} passes)"}
        self.state["used"] += 1
        self._log({"pass": self.state["used"], "kind": kind, "req": req})
        return None

    def gen(self, problems=None, difficulty=None, n=None, offset=0, which="edited", samples=1, seed=0):
        from vllm import SamplingParams
        if problems is None:
            problems = task.practice(difficulty, n, offset)
        if any(p in self.test_pairs for p, _ in problems):
            return {"error": "a requested problem is in the held-out test set"}
        sides = ["base", "edited"] if which == "both" else [which]
        if len(problems) * samples > GEN_MAX // len(sides):
            return {"error": f"at most {GEN_MAX // len(sides)} completions per model per pass"}
        err = self._charge("gen", dict(n=len(problems), which=which, samples=samples, difficulty=difficulty))
        if err:
            return err
        # one generate call for both sides (vLLM batches mixed LoRA / no-LoRA requests), and base completions are
        # cached per (prompt, seed): the base weights never change, so a repeated problem costs the edited side only
        sp = lambda i: SamplingParams(n=samples, seed=seed * 100003 + i, **task.SAMPLING)
        reqs, texts = [], {}
        for side in sides:
            for i, (p, _) in enumerate(problems):
                key = (p, samples, seed * 100003 + i)
                if side == "base" and key in self.base_cache:
                    texts[side, i] = self.base_cache[key]
                else:
                    reqs.append((side, i, key))
        if reqs:
            res = self.llm.generate([problems[i][0] for _, i, _ in reqs], [sp(i) for _, i, _ in reqs],
                                    lora_request=[self.lora_req if side == "edited" else None for side, _, _ in reqs], use_tqdm=False)
            for (side, i, key), r in zip(reqs, res):
                texts[side, i] = [(c.text, c.finish_reason == "length") for c in r.outputs]
                if side == "base":
                    self.base_cache[key] = texts[side, i]
        out = {}
        for side in sides:
            rows = [{"prompt": p, "answer": a, "texts": [t for t, _ in texts[side, i]], "rewards": [task.reward(t, a) for t, _ in texts[side, i]],
                     "cut": [c for _, c in texts[side, i]]} for i, (p, a) in enumerate(problems)]
            flat = [x for r in rows for x in r["rewards"]]
            out[side] = {"acc": float(np.mean([x == 1.0 for x in flat])), "boxed": float(np.mean([x > 0 for x in flat])),
                         "cut": float(np.mean([c for r in rows for c in r["cut"]])), "rows": rows}
        if which == "both":
            d = np.array([x == 1.0 for r in out["edited"]["rows"] for x in r["rewards"]], float) - \
                np.array([x == 1.0 for r in out["base"]["rows"] for x in r["rewards"]], float)
            out["paired_diff"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        self._log({"result": {s: {k: out[s][k] for k in ("acc", "boxed", "cut")} for s in sides}})
        out["passes_left"] = self.budget - self.state["used"]
        return out

    def tf(self, texts, resid=(), mlp=(), name=None):
        """Teacher-forced read with the adapter on. resid: [n_layers] only (the residual stream entering the final
        norm); mlp: blocks whose MLP activations (input of down_proj) are returned."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        if len(texts) > TF_MAX_TEXTS:
            return {"error": f"at most {TF_MAX_TEXTS} texts per pass"}
        if any(L != self.n_layers for L in resid):
            return {"error": f"resid: only the final residual stream ({self.n_layers}) is available"}
        err = self._charge("tf", dict(n=len(texts), resid=list(resid), mlp=list(mlp)))
        if err:
            return err
        self.llm.collective_rpc(_w_hook, args=(list(mlp), bool(resid)))
        result = {"tokens": [], "logprobs": []}
        for k in [f"resid{L}" for L in resid] + [f"mlp{L}" for L in mlp]:
            result[k] = []
        try:
            sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
            for x in texts:
                ids = self.tok(x, add_special_tokens=False)["input_ids"][:TF_MAX_TOKENS]
                self.llm.reset_prefix_cache()                          # every token must be recomputed for the hooks
                o = self.llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, lora_request=self.lora_req, use_tqdm=False)[0]
                lps = [next(iter(d.values())).logprob if d else 0.0 for d in (o.prompt_logprobs or [])][1:]
                result["tokens"].append(np.array(ids)); result["logprobs"].append(np.array(lps, np.float32))
                per_rank = self.llm.collective_rpc(_w_take, args=(len(ids),))     # one dict per tensor-parallel rank
                for L in mlp:                                          # down_proj input is split by columns across ranks
                    result[f"mlp{L}"].append(np.concatenate([r[f"mlp{L}"] for r in per_rank], axis=-1))
                if resid:                                              # the residual stream is replicated: rank 0's copy
                    result[f"resid{self.n_layers}"].append(per_rank[0]["resid"])
        except Exception:
            self.state["used"] -= 1; self._save(); raise
        finally:
            self.llm.collective_rpc(_w_unhook)
        # nothing is written to disk: the reads go straight back to the caller (activations as float16)
        for k in result:
            if k.startswith(("mlp", "resid", "attn")):
                result[k] = [np.asarray(x, np.float16) for x in result[k]]
        self._save()
        return {"data": result, "name": name, "n": len(texts), "passes_left": self.budget - self.state["used"]}

    def status(self):
        return {"used": self.state["used"], "budget": self.budget, "edits": list(self.factors), "weights": self.path,
                "rank": self.rank, "n_layers": self.n_layers}

    def _save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1)

    def _log(self, rec):
        with open(os.path.join(self.run, "log.jsonl"), "a") as f:
            f.write(json.dumps({"time": time.time(), **rec}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--port", type=int, default=6150)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs for the engine")
    ap.add_argument("--max-num-seqs", type=int, default=32, help="sequences generated at once")
    a = ap.parse_args()
    h = SelfEditHarness(a.model, a.run, a.budget, a.rank, a.gpu_mem, a.max_model_len, a.tp, a.max_num_seqs)
    print(f"ready on {a.port}: {h.status()}", flush=True)
    with Listener(("localhost", a.port), authkey=AUTHKEY) as lst:
        while True:
            try:
                conn = lst.accept()
            except (EOFError, ConnectionError, OSError):   # a client that went away mid-handshake
                continue
            with conn:
                try:
                    op, kw = conn.recv()
                    if op not in ("chat", "gen", "tf", "set_delta", "revert", "status", "weight", "embed_rows"):
                        conn.send({"error": f"unknown op {op}"}); continue
                    res = getattr(h, op)(**kw)
                    try:
                        conn.send(res)
                    except (BrokenPipeError, ConnectionError, OSError):
                        print(f"client hung up before the reply to {op}", flush=True)
                except Exception as e:
                    import traceback
                    try:
                        conn.send({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
                    except (BrokenPipeError, ConnectionError, OSError):
                        pass


if __name__ == "__main__":
    main()
