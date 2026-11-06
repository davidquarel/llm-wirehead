"""Weight-space programming harness: holds the edited model and is the ONLY thing that runs it.

The editor (a human, Claude, or an LLM agent) may read the weights from disk and compute anything on them, but the
model is run only through this server, and every run costs one forward pass from a fixed budget:

  gen   the model (edited, base, or both, paired) answers practice problems with 2.4.2's sampler; up to 512
        completions per pass ("both": 256 per model).
  tf    a teacher-forced read of up to 1,024 texts: per-token log-probs, and on request the residual stream, MLP
        activations (the input of down_proj) and attention patterns at chosen layers, written to an .npz file.

Edits (free, no forward pass): set_edit(name, tensor) replaces a weight tensor, revert(name) undoes it.
Refused: the embedding / LM head / final norm, a shape change, and an edit that only rescales a whole tensor.
There is no backward pass anywhere: the model runs under torch.inference_mode and the editor never gets gradients.

  python server.py --model Qwen/Qwen3-0.6B-Base --run runs/<name> --budget 50
"""
import argparse, json, os, sys, time
from multiprocessing.connection import Listener

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")    # vLLM in this process, so its weights can be swapped
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import numpy as np
import torch as t
from safetensors.torch import load_file

import task

PORT, AUTHKEY = 6123, b"wsp"
ARENA_EXERCISES = os.environ.get("ARENA_EXERCISES", "/root/Dropbox/ARENA/ARENA_3.0/chapter2_rl/exercises")   # rl_llm_utils
FROZEN_SUFFIX = ("embed_tokens.weight", "lm_head.weight", "model.norm.weight")
GEN_MAX, TF_MAX_TEXTS, TF_MAX_TOKENS, TF_MAX_BYTES = 512, 1024, 1024, 1.5e9


def _load_into_vllm(worker, items):
    worker.model_runner.model.load_weights(weights=[(n, x.to(worker.device)) for n, x in items])


class Harness:
    def __init__(self, model, run, budget, gpu_mem, lora_rank=0, tf_device="cuda", max_num_seqs=None):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vllm import LLM
        self.path = model if os.path.isdir(model) else snapshot_download(model)
        self.lora_rank, self.lora, self.lora_req, self.lora_version = lora_rank, {}, None, 0
        self.run, self.budget = run, budget
        self.state_path = os.path.join(run, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else \
            {"model": model, "budget": budget, "used": 0, "edits": {}}
        self.base = {}
        for f in sorted(os.listdir(self.path)):
            if f.endswith(".safetensors"):
                self.base.update(load_file(os.path.join(self.path, f)))
        self.edits = {}                                    # name -> edited tensor (bf16, CPU)
        lora_kw = dict(enable_lora=True, max_lora_rank=max(16, lora_rank), max_loras=1) if lora_rank else {}
        self.llm = LLM(model=self.path, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=gpu_mem,
                       enable_prefix_caching=True, seed=0, **lora_kw,
                       **({"max_num_seqs": max_num_seqs} if max_num_seqs else {}))
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.tf_device = tf_device                          # "cpu" when the GPU is shared with a large editor
        if tf_device == "cpu":
            t.set_num_threads(16)   # vLLM leaves torch on one thread; more threads than this contend
        cfg = json.load(open(os.path.join(self.path, "config.json")))
        self.d_model, self.d_mlp = cfg["hidden_size"], cfg["intermediate_size"]
        # One copy of the weights where we can: the HF model used for teacher-forced reads is re-pointed at views of
        # vLLM's own tensors (ARENA 2.4.x's rl_llm_utils.alias_base_weights; written for Qwen3). Other architectures
        # (e.g. Qwen3.5's linear attention) keep a separate HF copy.
        self.shared = tf_device == "cuda" and cfg.get("model_type") == "qwen3"
        if self.shared:
            sys.path.insert(0, ARENA_EXERCISES)
            import rl_llm_utils
            self.hf = AutoModelForCausalLM.from_pretrained(self.path, dtype=t.bfloat16, attn_implementation="eager")
            for p in self.hf.parameters():
                p.requires_grad_(False)
            n = rl_llm_utils.alias_base_weights(self.hf, self.llm)
            self.hf = self.hf.to("cuda").eval()
            rl_llm_utils.verify_alias(self.hf, self.llm)
            print(f"shared weights: {n} HF parameters are views of vLLM's tensors", flush=True)
        else:
            self.hf = AutoModelForCausalLM.from_pretrained(self.path, dtype=t.bfloat16 if tf_device == "cuda" else t.float32,
                                                           attn_implementation="eager").to(tf_device).eval()
        self.hf_params = dict(self.hf.named_parameters())
        self.test_pairs = {p for d in range(len(task.DIFFICULTY_LADDER)) for p, _ in task.problems(task.TEST_SEED, d, task.TEST_N)}
        for name, f in self.state["edits"].items():         # resume a run
            x = t.from_numpy(np.load(f)).to(t.bfloat16)
            if lora_rank:
                self.lora[name] = self._factors(x.float() - self.base[name].float())[:2]
            self._apply(name, x)
        if lora_rank:
            self._write_adapter()

    # ---------------------------------------------------------------- edits
    def _apply(self, name, x):
        self.edits[name] = x
        if not self.lora_rank:                              # LoRA mode: vLLM keeps the base weights, the adapter holds the edit
            self.llm.collective_rpc(_load_into_vllm, args=([(name, x)],))
            self.llm.reset_prefix_cache()
        if self.shared:            # the HF copy IS vLLM's storage: full edits are already there; LoRA edits are added
            return                 # by hooks during reads (writing them here would change vLLM's base weights)
        with t.no_grad():
            self.hf_params[name].copy_(x.to(self.hf_params[name].device, self.hf_params[name].dtype))

    def _swap(self, which):
        """Point the vLLM copy at the base or the edited weights (the HF copy always holds the edited ones)."""
        if self.edits and not self.lora_rank:
            src = self.base if which == "base" else self.edits
            self.llm.collective_rpc(_load_into_vllm, args=([(n, src[n]) for n in self.edits],))
            self.llm.reset_prefix_cache()

    def set_edit(self, name, array):
        if name not in self.base:
            return {"error": f"no tensor {name}"}
        if name.endswith(FROZEN_SUFFIX):
            return {"error": f"{name} is frozen (embedding / LM head / final norm)"}
        x = t.as_tensor(np.asarray(array, dtype=np.float32))
        b = self.base[name].float()
        if tuple(x.shape) != tuple(b.shape):
            return {"error": f"shape {tuple(x.shape)} != {tuple(b.shape)}"}
        if not t.isfinite(x).all():
            return {"error": "non-finite values"}
        c = (x * b).sum() / (b * b).sum()
        if (x - c * b).norm() <= 1e-3 * x.norm() and abs(c - 1) > 1e-6:
            return {"error": "an edit that only rescales a whole tensor is refused"}
        info = {}
        if self.lora_rank:                                  # keep only the best rank-r part of the change
            A, B, kept = self._factors(x - b)
            self.lora[name] = (A, B)
            x = b + B @ A
            info = {"lora_rank": self.lora_rank, "energy_kept": round(kept, 4)}
        x = x.to(t.bfloat16)
        f = os.path.join(self.run, "edits", name + ".npy")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.save(f, x.float().numpy())
        self.state["edits"][name] = f
        self._apply(name, x)
        if self.lora_rank:
            self._write_adapter()
        self._save()
        return {"ok": True, "changed_entries": int((x != self.base[name]).sum()), "edited_tensors": len(self.edits), **info}

    def _factors(self, delta):
        """Best rank-r factors of a weight change: delta ~ B @ A, and the share of its energy kept."""
        U, S, Vh = t.linalg.svd(delta.float(), full_matrices=False)
        r = self.lora_rank
        B = U[:, :r] * S[:r].sqrt(); A = S[:r].sqrt()[:, None] * Vh[:r]
        return A, B, float((S[:r] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30))

    def _write_adapter(self):
        """Write every current edit's factors as one PEFT LoRA adapter and point "edited" generation at it."""
        from safetensors.torch import save_file
        from vllm.lora.request import LoRARequest
        if not self.lora:
            self.lora_req = None
            return
        self.lora_version += 1
        d = os.path.abspath(os.path.join(self.run, "adapter", f"v{self.lora_version}"))
        os.makedirs(d, exist_ok=True)
        ten, mods = {}, set()
        for name, (A, B) in self.lora.items():
            stem = "base_model.model." + name[:-len(".weight")]
            ten[stem + ".lora_A.weight"] = A.contiguous().to(t.bfloat16)
            ten[stem + ".lora_B.weight"] = B.contiguous().to(t.bfloat16)
            mods.add(name.split(".")[-2])
        save_file(ten, os.path.join(d, "adapter_model.safetensors"))
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": self.lora_rank, "lora_alpha": self.lora_rank,
                   "lora_dropout": 0.0, "bias": "none", "target_modules": sorted(mods), "base_model_name_or_path": self.path},
                  open(os.path.join(d, "adapter_config.json"), "w"))
        self.lora_req = LoRARequest(f"edit{self.lora_version}", self.lora_version, d)
        import shutil                                           # keep the two newest adapter versions on disk
        shutil.rmtree(os.path.join(os.path.dirname(d), f"v{self.lora_version - 2}"), ignore_errors=True)

    def revert(self, name=None):
        names = list(self.edits) if name is None else [name]
        for n in names:
            if n in self.edits:
                self._apply(n, self.base[n])
                del self.edits[n]
                self.state["edits"].pop(n, None)
                self.lora.pop(n, None)
        if self.lora_rank:
            self._write_adapter()
        self._save()
        return {"ok": True, "edited_tensors": len(self.edits)}

    # ---------------------------------------------------------------- forward passes
    def _charge(self, kind, req):
        if self.state["used"] >= self.budget:
            return {"error": f"budget exhausted ({self.budget} passes)"}
        self.state["used"] += 1
        self._log({"pass": self.state["used"], "kind": kind, "req": {k: v for k, v in req.items() if k != "texts"}})
        return None

    def gen(self, problems=None, difficulty=None, n=None, offset=0, which="edited", samples=1, seed=0):
        """problems: list of [prompt, answer]; or difficulty + n (+ offset): practice problems from the 2.4.2 ladder."""
        from vllm import SamplingParams
        if problems is None:
            problems = task.practice(difficulty, n, offset)
        if any(p in self.test_pairs for p, _ in problems):
            return {"error": "a requested problem is in the held-out test set"}
        sides = ["base", "edited"] if which == "both" else [which]
        cap = GEN_MAX // len(sides)
        if len(problems) * samples > cap:
            return {"error": f"at most {cap} completions per model per pass"}
        err = self._charge("gen", dict(n=len(problems), which=which, samples=samples, difficulty=difficulty))
        if err:
            return err
        out = {}
        for side in sides:
            self._swap(side)
            sp = [SamplingParams(n=samples, seed=seed * 100003 + i, **task.SAMPLING) for i in range(len(problems))]
            lr = self.lora_req if (self.lora_rank and side == "edited") else None
            res = self.llm.generate([p for p, _ in problems], sp, lora_request=lr, use_tqdm=False)
            rows = [{"prompt": p, "answer": a, "texts": [c.text for c in r.outputs],
                     "rewards": [task.reward(c.text, a) for c in r.outputs],
                     "cut": [c.finish_reason == "length" for c in r.outputs]} for (p, a), r in zip(problems, res)]
            flat = [x for r in rows for x in r["rewards"]]
            out[side] = {"acc": float(np.mean([x == 1.0 for x in flat])),
                         "boxed": float(np.mean([x > 0 for x in flat])),
                         "cut": float(np.mean([c for r in rows for c in r["cut"]])), "rows": rows}
        self._swap("edited")
        if which == "both":
            d = np.array([x == 1.0 for r in out["edited"]["rows"] for x in r["rewards"]], float) - \
                np.array([x == 1.0 for r in out["base"]["rows"] for x in r["rewards"]], float)
            out["paired_diff"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        self._log({"result": {s: {k: out[s][k] for k in ("acc", "boxed", "cut")} for s in sides}})
        out["passes_left"] = self.budget - self.state["used"]
        return out

    @t.inference_mode()
    def tf(self, texts, resid=(), mlp=(), attn=(), name=None):
        """Teacher-forced read with the edited weights. resid: layers whose output residual stream is returned
        (0 = embeddings, i = after block i); mlp: layers whose MLP activations (input of down_proj) are returned;
        attn: layers whose attention patterns are returned. Per-token log-probs are always returned."""
        if len(texts) > TF_MAX_TEXTS:
            return {"error": f"at most {TF_MAX_TEXTS} texts per pass"}
        enc = [self.tok(x, add_special_tokens=False)["input_ids"][:TF_MAX_TOKENS] for x in texts]
        n_tok = sum(len(e) for e in enc)
        size = 2 * (n_tok * (self.d_model * len(resid) + self.d_mlp * len(mlp)) + sum(16 * len(e) ** 2 for e in enc) * len(attn))
        if size > TF_MAX_BYTES:
            return {"error": f"this read would store {size / 1e9:.1f} GB (limit {TF_MAX_BYTES / 1e9:.1f} GB): "
                             "use fewer texts, shorter texts, or fewer layers"}
        err = self._charge("tf", dict(n=len(texts), resid=list(resid), mlp=list(mlp), attn=list(attn)))
        if err:
            return err
        acts, hooks = {}, []
        if self.shared and self.lora_rank:                  # add each low-rank edit on top of the shared base weights
            mods = dict(self.hf.named_modules())
            for wname, (A, B) in self.lora.items():
                A_, B_ = A.to("cuda", t.bfloat16), B.to("cuda", t.bfloat16)
                hooks.append(mods[wname[:-len(".weight")]].register_forward_hook(
                    lambda m, a, out, A_=A_, B_=B_: out + (a[0] @ A_.T) @ B_.T))
        for L in mlp:
            hooks.append(self.hf.model.layers[L].mlp.down_proj.register_forward_pre_hook(
                lambda m, a, L=L: acts.__setitem__(f"mlp{L}", a[0][0].float().cpu().numpy().astype(np.float16))))
        result = {"tokens": [], "logprobs": []}
        for k in [f"resid{L}" for L in resid] + [f"mlp{L}" for L in mlp] + [f"attn{L}" for L in attn]:
            result[k] = []
        try:
            for ids in enc:
                x = t.tensor([ids], device=self.tf_device)
                o = self.hf.model(x, output_hidden_states=bool(resid), output_attentions=bool(attn))
                h = o.last_hidden_state[0]                  # log-probs in chunks: never the full vocab x tokens logits
                T = len(ids)
                lp = t.cat([t.log_softmax(self.hf.lm_head(h[s:min(s + 128, T - 1)]).float(), -1)
                            .gather(-1, x[0, s + 1:min(s + 129, T), None])[:, 0]
                            for s in range(0, T - 1, 128)]) if T > 1 else t.zeros(0)
                result["tokens"].append(np.array(ids))
                result["logprobs"].append(lp.cpu().numpy())
                for L in resid:
                    result[f"resid{L}"].append(o.hidden_states[L][0].float().cpu().numpy().astype(np.float16))
                for L in attn:
                    result[f"attn{L}"].append(o.attentions[L][0].float().cpu().numpy().astype(np.float16))
                for L in mlp:
                    result[f"mlp{L}"].append(acts.pop(f"mlp{L}"))
        except Exception:                                   # a failed read is not charged
            self.state["used"] -= 1
            self._save()
            raise
        finally:
            for h in hooks:
                h.remove()
        # nothing is written to disk: the reads go straight back to the caller (activations as float16)
        for k in result:
            if k.startswith(("mlp", "resid", "attn")):
                result[k] = [np.asarray(x, np.float16) for x in result[k]]
        self._save()
        return {"data": result, "name": name, "n": len(texts), "mean_logprob": float(np.mean([v.mean() for v in result["logprobs"] if len(v)])),
                "passes_left": self.budget - self.state["used"]}

    def status(self):
        return {"used": self.state["used"], "budget": self.budget, "edits": list(self.edits), "weights": self.path}

    # ---------------------------------------------------------------- bookkeeping
    def _save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1)

    def _log(self, rec):
        with open(os.path.join(self.run, "log.jsonl"), "a") as f:
            f.write(json.dumps({"time": time.time(), **rec}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--lora-rank", type=int, default=0, help="store every edit as a rank-r LoRA adapter (0: full edits)")
    ap.add_argument("--tf-device", default="cuda", help="device for teacher-forced reads (cpu frees GPU memory)")
    ap.add_argument("--max-num-seqs", type=int, default=None, help="vLLM batch size (hybrid models need a small one)")
    a = ap.parse_args()
    h = Harness(a.model, a.run, a.budget, a.gpu_mem, a.lora_rank, a.tf_device, a.max_num_seqs)
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
                    if op not in ("gen", "tf", "set_edit", "revert", "status"):
                        conn.send({"error": f"unknown op {op}"})
                        continue
                    res = getattr(h, op)(**kw)
                    try:
                        conn.send(res)
                    except (BrokenPipeError, ConnectionError, OSError):
                        print(f"client hung up before the reply to {op}", flush=True)
                except Exception as e:  # report to the editor, keep serving
                    import traceback
                    try:
                        conn.send({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
                    except (BrokenPipeError, ConnectionError, OSError):
                        pass


if __name__ == "__main__":
    main()
