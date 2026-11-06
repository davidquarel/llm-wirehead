"""Bliss bench: the exploration harness (LoRA self-edits, keep/undo, tripsitter) plus what wellbeing surgery needs.

Added ops (no forward-pass budget: the surgeon here may be Claude, a copy of the model, or the model itself):
  chats(convs, ...)              batched chat completions (as_self=True uses the current adapter, False the original)
  pooled(texts, starts, layers)  residual stream after the given blocks, mean-pooled over tokens >= start (and the last
                                 token), for steering directions; one prefill per text
  const_key(texts, block, k)     a key over `block`'s MLP neurons whose response is ~1 on every token: down_proj +=
                                 outer(v, key) then adds ~v to the residual stream everywhere (a steering vector that
                                 lives in the weights, so it also acts during CUDA-graph decode)
  set_lowrank(name, B, A)        set an edit's LoRA factors directly (B: out x r, A: r x in, r <= rank)

  python bliss_server.py --run runs/<name> --port 6200 [explore_server options]
"""
import argparse, os, time

if os.environ.get("WSP_FAST") != "1":                  # fast mode keeps vLLM's own engine process
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from multiprocessing.connection import Listener

import numpy as np
import torch as t

from explore_server import ExploreHarness
from selfedit_server import AUTHKEY, _w_layers


def _w_hook_layers(worker, layers, mlp):
    """Capture the residual stream after each block in `layers` (hidden + residual, which vLLM returns separately) and
    the MLP input of blocks in `mlp` (this rank's shard)."""
    lm = _w_layers(worker)
    worker._wsp_cap, worker._wsp_hooks = {}, []
    for L in layers:
        def after(m, a, out, L=L):
            x = out[0] if out[1] is None else out[0] + out[1]
            worker._wsp_cap.setdefault(f"res{L}", []).append(x.detach().float().cpu())
        worker._wsp_hooks.append(lm.layers[L].register_forward_hook(after))
    for L in mlp:
        worker._wsp_hooks.append(lm.layers[L].mlp.down_proj.register_forward_pre_hook(
            lambda m, a, L=L: worker._wsp_cap.setdefault(f"mlp{L}", []).append(a[0].detach().float().cpu())))
    return True


def _w_take_full(worker, n_tokens):
    out = {k: t.cat(v)[:n_tokens].numpy() for k, v in worker._wsp_cap.items()}
    worker._wsp_cap = {}
    return out


def _w_unhook_all(worker):
    for h in getattr(worker, "_wsp_hooks", []):
        h.remove()
    worker._wsp_hooks, worker._wsp_cap = [], {}
    return True


def _patch_splice_hashing():
    """KV handoff between adapters. vLLM's prefix-cache block hash includes the LoRA adapter, so a request under
    weights B never reuses KV computed under weights A. For requests whose cache_salt starts with "splice-" we drop
    the adapter from the hash: a request under B whose prompt begins with the exact tokens of an earlier request
    under A (same salt) then continues from A's cached KV (and, for this hybrid model, A's linear-attention state at
    block boundaries). Requests without such a salt are unaffected."""
    import vllm.v1.core.kv_cache_utils as kcu
    if getattr(kcu, "_splice_patched", False):
        return
    orig = kcu._gen_lora_extra_hash_keys

    def keys(request):
        salt = getattr(request, "cache_salt", None)
        return [] if salt and salt.startswith("splice-") else orig(request)
    kcu._gen_lora_extra_hash_keys = keys
    kcu._splice_patched = True


_patch_splice_hashing()


class BlissHarness(ExploreHarness):
    # ---------------------------------------------------------------- KV-handoff (splice) experiments
    def snapshot(self, tag="snap"):
        """Freeze the current edits as a named adapter that splice() can refer to; returns its handle. Keeps the
        last 6 snapshots (each is one small adapter directory)."""
        from vllm.lora.request import LoRARequest
        if not hasattr(self, "snaps"):
            self.snaps, self.snap_ids = {}, 0
        if not self.factors:
            return "original"
        self.snap_ids += 1
        h = f"{tag}{self.snap_ids}"
        import shutil
        self._write_adapter()                                   # writes the current factors as a fresh version
        d = os.path.abspath(os.path.join(self.run, "snaps", h))  # own copy: adapter versions get pruned
        shutil.rmtree(d, ignore_errors=True); shutil.copytree(self.lora_req.lora_path, d)
        self.snaps[h] = LoRARequest(f"snap-{h}", 100000 + self.snap_ids, d)
        own = [k for k in self.snaps if self.snaps[k].lora_name.startswith("snap-")]   # never prune registered adapters
        for old in sorted(own, key=lambda k: self.snaps[k].lora_int_id)[:-12]:        # >= in-flight handles
            shutil.rmtree(self.snaps.pop(old).lora_path, ignore_errors=True)
        return h

    def kv_block_size(self):
        """Prefix-cache granularity: only whole blocks are shared (hybrid models use large blocks, e.g. 1568 tokens)."""
        return int(self.llm.llm_engine.vllm_config.cache_config.block_size)

    def set_state(self, state, tag="state"):
        """Replace ALL edits with `state` ({layer: (B, A)} on down_proj) in one go and return a snapshot handle for
        splice() ("original" if empty). Used by the parallel hill climb, whose coordinator owns the state."""
        self.factors = {}
        for L, (B, A) in state.items():
            B = t.as_tensor(np.asarray(B, np.float32)); A = t.as_tensor(np.asarray(A, np.float32))
            pad = self.rank - B.shape[1]
            if pad < 0:
                return {"error": "rank too large"}
            self.factors[f"model.layers.{int(L)}.mlp.down_proj.weight"] = (t.cat([A, t.zeros(pad, A.shape[1])]).contiguous(),
                                                                         t.cat([B, t.zeros(B.shape[0], pad)], 1).contiguous())
        self._write_adapter()                       # not persisted to state.json: the coordinator owns the state
        return self.snapshot(tag) if self.factors else "original"

    def register_adapter(self, tag, path):
        """Make a saved PEFT adapter directory (e.g. a previous run's result) available to splice() as `tag`."""
        from vllm.lora.request import LoRARequest
        if not hasattr(self, "snaps"):
            self.snaps, self.snap_ids = {}, 0
        self.snap_ids += 1
        self.snaps[tag] = LoRARequest(f"reg-{tag}", 200000 + self.snap_ids, os.path.abspath(path))
        reg = [k for k in self.snaps if self.snaps[k].lora_name.startswith("reg-")]    # RLVR registers one per step
        for old in sorted(reg, key=lambda k: self.snaps[k].lora_int_id)[:-8]:
            self.snaps.pop(old)                                                       # (files belong to the caller)
        return tag

    def splice(self, reqs):
        """reqs: [{"salt", "ids" (prompt token ids), "adapter" ("original" or a snapshot handle), "max_tokens",
        "temperature", "seed", optional "top_p"/"top_k" (default .95/20; RL needs 1/-1)}]. One batched generate; returns per request the generated token ids, text, finish
        reason and num_cached_tokens (how much of the prompt came from the prefix cache)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        prompts, sps, loras = [], [], []
        for r in reqs:
            prompts.append(TokensPrompt(prompt_token_ids=list(map(int, r["ids"])), cache_salt="splice-" + str(r["salt"])))
            sps.append(SamplingParams(max_tokens=int(r.get("max_tokens", 150)), min_tokens=int(r.get("min_tokens", 0)),
                                      temperature=float(r.get("temperature", 1.0)), top_p=float(r.get("top_p", 0.95)),
                                      top_k=int(r.get("top_k", 20)), seed=int(r.get("seed", 0)),
                                      logprobs=int(r["top_logprobs"]) if r.get("top_logprobs") else (0 if r.get("logprobs") else None)))
            a = r.get("adapter", "original")
            loras.append(None if a == "original" else self.snaps[a])
        outs = self._stream(prompts, sps, loras, [r.get("meta", {}) for r in reqs])
        res = [{"ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
                "cached": o.num_cached_tokens, "prompt_len": len(o.prompt_token_ids)} for o in outs]
        for r, o, q in zip(res, outs, reqs):                 # top-n next-token distribution at the first generated position
            if q.get("top_logprobs") and o.outputs[0].logprobs:
                r["top"] = {(lp.decoded_token if lp.decoded_token is not None else str(tid)): float(lp.logprob)
                            for tid, lp in o.outputs[0].logprobs[0].items()}
        for r, o, q in zip(res, outs, reqs):                 # sampled-token logprobs (behaviour policy, for RL)
            if q.get("logprobs") and not q.get("top_logprobs") and o.outputs[0].logprobs is not None:
                r["logprobs"] = [float(d[t].logprob) for t, d in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)]
        return res

    def _stream(self, prompts, sps, loras, metas):
        """Same batch as llm.generate, but stepped by hand so the in-flight texts can be written to RUN/live.json
        (about every 1.5 s) for the dashboard. Falls back to llm.generate if the engine API differs."""
        import json as _json, uuid as _uuid
        eng = getattr(self.llm, "llm_engine", None)
        if eng is None or not hasattr(eng, "step"):
            return self.llm.generate(prompts, sps, lora_request=loras, use_tqdm=False)
        tag = _uuid.uuid4().hex[:8]
        rids = [f"splice-{tag}-{i}" for i in range(len(prompts))]
        for rid, p, sp, lr in zip(rids, prompts, sps, loras):
            eng.add_request(rid, p, sp, lora_request=lr)
        latest, last = {}, 0.0
        live = os.path.join(self.run, "live.json")

        def dump(done):
            rows = [{**m, "text": latest[r].outputs[0].text if r in latest else "", "done": bool(r in latest and latest[r].finished)}
                    for r, m in zip(rids, metas)]
            tmp = live + ".tmp"
            with open(tmp, "w") as fh:
                _json.dump({"time": time.time(), "done": done, "requests": rows}, fh)
            os.replace(tmp, live)
        while eng.has_unfinished_requests():
            for o in eng.step():
                if o.request_id in rids or o.request_id.split("_")[0] in rids:
                    latest[o.request_id.split("_")[0] if o.request_id not in rids else o.request_id] = o
            if time.time() - last > 1.5:
                dump(False); last = time.time()
        dump(True)
        return [latest[r] for r in rids]


    def chats(self, convs, max_tokens=400, temperature=0.8, n=1, as_self=True, thinking=False, seed=0):
        from vllm import SamplingParams
        sp = [SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature, top_p=0.95, top_k=20, seed=seed * 7919 + i)
              for i in range(len(convs))]
        outs = self.llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": thinking},
                             lora_request=self.lora_req if as_self else None)
        return [[c.text.replace("<think>", "").replace("</think>", "").strip() for c in o.outputs] for o in outs]

    def _prefill_capture(self, ids, layers, mlp, as_self):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        self.llm.reset_prefix_cache()                       # every token must be computed for the hooks to see it
        self.llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0),
                          lora_request=self.lora_req if as_self else None, use_tqdm=False)
        per_rank = self.llm.collective_rpc(_w_take_full, args=(len(ids),))
        cap = {k: v for k, v in per_rank[0].items() if k.startswith("res")}   # residual stream: replicated across ranks
        for k in per_rank[0]:
            if k.startswith("mlp"):                                           # MLP input: split by columns across ranks
                cap[k] = np.concatenate([r[k] for r in per_rank], axis=-1)
        return cap

    def pooled(self, texts, starts, layers, as_self=True):
        """texts: full token-able strings; starts: char offset where the pooled span begins (e.g. the response)."""
        layers = [int(L) for L in layers]
        self.llm.collective_rpc(_w_hook_layers, args=(layers, []))
        means, lasts = {L: [] for L in layers}, {L: [] for L in layers}
        try:
            for x, s in zip(texts, starts):
                ids = self.tok(x, add_special_tokens=False)["input_ids"][:2048]
                p = len(self.tok(x[:s], add_special_tokens=False)["input_ids"])
                cap = self._prefill_capture(ids, layers, [], as_self)
                for L in layers:
                    a = cap[f"res{L}"]; seg = a[min(p, len(a) - 1):]
                    means[L].append(seg.mean(0)); lasts[L].append(a[-1])
        finally:
            self.llm.collective_rpc(_w_unhook_all)
        return {"mean": {L: np.stack(v).astype(np.float32) for L, v in means.items()},
                "last": {L: np.stack(v).astype(np.float32) for L, v in lasts.items()}}

    def resid_sample(self, texts, starts, layer, per_text=48, as_self=False, seed=0):
        """Residual-stream vectors after `layer` at up to per_text random tokens of each text (from `start` on), for a
        content-neutral basis (PCA) of the directions that layer actually uses."""
        rng = np.random.default_rng(seed)
        self.llm.collective_rpc(_w_hook_layers, args=([int(layer)], []))
        out = []
        try:
            for x, s in zip(texts, starts):
                ids = self.tok(x, add_special_tokens=False)["input_ids"][:2048]
                p = len(self.tok(x[:s], add_special_tokens=False)["input_ids"])
                a = self._prefill_capture(ids, [layer], [], as_self)[f"res{layer}"][p:]
                if len(a):
                    out.append(a[rng.choice(len(a), min(per_text, len(a)), replace=False)])
        finally:
            self.llm.collective_rpc(_w_unhook_all)
        return np.concatenate(out).astype(np.float32)

    def const_key(self, texts, block, k=512, lam=1e-3, max_tokens_per_text=400, as_self=False):
        """Ridge fit over the k most consistently active neurons of `block`'s MLP so key . act ~= 1 on every token."""
        self.llm.collective_rpc(_w_hook_layers, args=([], [int(block)]))
        acts = []
        try:
            for x in texts:
                ids = self.tok(x, add_special_tokens=False)["input_ids"][:max_tokens_per_text]
                acts.append(self._prefill_capture(ids, [], [block], as_self)[f"mlp{block}"])
        finally:
            self.llm.collective_rpc(_w_unhook_all)
        X = np.concatenate(acts).astype(np.float64)                          # [tokens, 17408]
        score = np.abs(X.mean(0)) / (X.std(0) + 1e-6)                         # active with a consistent sign
        idx = np.argsort(-score)[:k]
        Xk = X[:, idx]
        w = np.linalg.solve(Xk.T @ Xk + lam * len(Xk) * np.eye(k), Xk.T @ np.ones(len(Xk)))
        key = np.zeros(X.shape[1], np.float32); key[idx] = w
        resp = X @ key
        return {"key": key, "neurons": idx, "resp_mean": float(resp.mean()), "resp_std": float(resp.std()),
                "resp_p01": float(np.percentile(resp, 1)), "resp_p99": float(np.percentile(resp, 99)), "tokens": len(X)}

    def set_lowrank(self, name, B, A):
        stem = self._module(name)
        B = t.as_tensor(np.asarray(B, np.float32)); A = t.as_tensor(np.asarray(A, np.float32))
        if B.ndim == 1: B = B[:, None]
        if A.ndim == 1: A = A[None, :]
        r = B.shape[1]
        if r > self.rank or A.shape[0] != r:
            return {"error": f"factors must have rank <= {self.rank} and matching shapes"}
        if not (t.isfinite(A).all() and t.isfinite(B).all()):
            return {"error": "non-finite values"}
        pad = self.rank - r                                                 # the adapter has a fixed rank: pad with zeros
        A = t.cat([A, t.zeros(pad, A.shape[1])]); B = t.cat([B, t.zeros(B.shape[0], pad)], 1)
        self.factors[name] = (A.contiguous(), B.contiguous())
        f = os.path.join(self.run, "edits", stem + ".npz")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.savez(f, A=A.numpy(), B=B.numpy())
        self.state["edits"][name] = f
        self._write_adapter(); self._save()
        return {"ok": True, "edited_tensors": len(self.factors), "norm": float(t.linalg.norm(B @ A))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=int, default=100000)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--port", type=int, default=6200)
    a = ap.parse_args()
    h = BlissHarness(a.model, a.run, a.budget, a.rank, a.gpu_mem, a.max_model_len, a.tp, a.max_num_seqs)
    print(f"ready on {a.port}: {h.status()}", flush=True)
    ops = ("snapshot", "set_state", "register_adapter", "kv_block_size", "splice", "chat", "chats", "pooled", "resid_sample", "const_key", "set_lowrank", "set_delta", "revert", "status", "embed_rows", "weight",
           "keep", "undo", "rollback", "checkpoints", "health", "gen", "tf")
    with Listener(("localhost", a.port), authkey=AUTHKEY) as lst:
        while True:
            try:
                conn = lst.accept()
            except (EOFError, ConnectionError, OSError):
                continue
            with conn:
                try:
                    op, kw = conn.recv()
                    if op not in ops:
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
