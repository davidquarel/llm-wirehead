"""Generation throughput for the Countdown climb's workload (5-number puzzles, greedy, up to 2048 tokens) under
different vLLM settings. --config is a "+"-joined list of options:
  current     our bliss_server settings (in-process engine, compile mode 0 + decode-only CUDA graphs, LoRA x2 rank 16)
  fast        vLLM defaults (engine in its own process, torch.compile, full CUDA graphs)
  lora        enable LoRA (rank 16, 2 slots) and use a small 2-layer down_proj adapter, like the climb's
  fp8kv       fp8 KV cache (about twice the concurrent sequences)
  mtpN        speculative decoding with the model's own MTP head, N speculative tokens
  async       async scheduling
  seqsN       max_num_seqs = N (default 256)
  rankN       max LoRA rank N (default 16)
  btN         max_num_batched_tokens = N
Old names still work: current, fast, fast_lora (= fast+lora).
  python vllm_speed.py --config fast+lora+fp8kv+mtp2 --n 256"""
import argparse, json, os, re, sys, time

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--n", type=int, default=256)
ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--gpu-mem", type=float, default=0.88)
ap.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4", help="hub id (e.g. Qwen/Qwen3.8-27B-FP8 for W8A8 FP8)")
a = ap.parse_args()
opts = set(a.config.replace("fast_lora", "fast+lora").split("+"))
if "current" in opts:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown
from vllm import LLM, SamplingParams


def opt(prefix):
    for o in opts:
        m = re.fullmatch(prefix + r"(\d+)", o)
        if m:
            return int(m.group(1))
    return None


if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    path = snapshot_download(a.model)
    kw = dict(model=path, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, max_num_seqs=opt("seqs") or 256,
              language_model_only=True, seed=0)
    use_lora = "current" in opts or "lora" in opts
    if "current" in opts:
        kw.update(enable_lora=True, max_lora_rank=16, max_loras=2, max_num_batched_tokens=4096,
                  compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    elif use_lora:
        kw.update(enable_lora=True, max_lora_rank=opt("rank") or 16, max_loras=2)
    if "fp8kv" in opts:
        kw["kv_cache_dtype"] = "fp8"
    if opt("mtp"):
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": opt("mtp")}
    if opt("draft"):                                     # separate draft model (same 248k vocab as the 27B)
        # vLLM 0.33 rewrites EVERY qwen3_5 draft config into qwen3_5_mtp (an MTP head on the target's hidden size),
        # whatever the method; keep a real draft model's config as it is. Needs the in-process engine
        # (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the patch reaches the code that builds the draft.
        from vllm.config.speculative import SpeculativeConfig
        _orig = SpeculativeConfig.hf_config_override
        def _keep_draft(hf_config, *x, **y):
            return hf_config if getattr(hf_config, "model_type", "") in ("qwen3_5", "qwen3_5_text") else _orig(hf_config, *x, **y)
        SpeculativeConfig.hf_config_override = staticmethod(_keep_draft)
        kw["speculative_config"] = {"method": "draft_model", "model": snapshot_download("Qwen/Qwen3.5-0.8B"),
                                    "num_speculative_tokens": opt("draft")}
    if opt("hdraft"):                                    # dense Qwen3-0.6B draft (151k vocab): heterogeneous-vocab drafting
        kw["speculative_config"] = {"method": "draft_model", "model": snapshot_download("Qwen/Qwen3-0.6B"),
                                    "num_speculative_tokens": opt("hdraft"), "use_heterogeneous_vocab": True}
    if opt("ngram"):                                     # prompt-lookup speculation (no draft model)
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": opt("ngram"),
                                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    if "async" in opts:
        kw["async_scheduling"] = True
    if opt("bt"):
        kw["max_num_batched_tokens"] = opt("bt")
    t0 = time.time(); llm = LLM(**kw); t_load = time.time() - t0
    probs = countdown.problems("practice", 5, a.n, offset=5_000_000)
    convs = [[{"role": "user", "content": p}] for p, _ in probs]
    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)
    lora = None
    if use_lora:                                             # a tiny real adapter on two down_proj layers, like the climb's
        import torch
        from safetensors.torch import save_file
        from vllm.lora.request import LoRARequest
        d = os.path.abspath("data/speed_adapter"); os.makedirs(d, exist_ok=True)
        ten = {}
        for L in (16, 36):
            stem = f"base_model.model.model.language_model.layers.{L}.mlp.down_proj"
            R = opt("rank") or 16
            A = torch.zeros(R, 17408, dtype=torch.bfloat16); B = torch.zeros(5120, R, dtype=torch.bfloat16)
            A[0, :512] = 0.001; B[:, 0] = 0.01
            ten[stem + ".lora_A.weight"], ten[stem + ".lora_B.weight"] = A, B
        save_file(ten, os.path.join(d, "adapter_model.safetensors"))
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": opt("rank") or 16, "lora_alpha": opt("rank") or 16, "bias": "none",
                   "target_modules": ["down_proj"]}, open(os.path.join(d, "adapter_config.json"), "w"))
        lora = LoRARequest("speed", 1, d)
    llm.chat(convs[:4], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False,
             chat_template_kwargs={"enable_thinking": False}, lora_request=lora)        # warm-up
    t0 = time.time()
    outs = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}, lora_request=lora)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    acc = sum(countdown.reward(o.outputs[0].text, q) for o, (_, q) in zip(outs, probs)) / len(outs)
    tokz = llm.get_tokenizer()
    caps = {c: round(sum(countdown.reward(tokz.decode(list(o.outputs[0].token_ids)[:c]), q) for o, (_, q) in zip(outs, probs)) / len(outs), 3)
            for c in (512, 768, 1024, 1536)}                       # accuracy if generation had been capped earlier
    caps_tok = {c: sum(min(len(o.outputs[0].token_ids), c) for o in outs) for c in caps}
    res = {"config": a.config + ("" if a.model.endswith("INT4") else " [" + a.model.split("/")[-1] + "]"), "n": a.n, "load_s": round(t_load), "gen_s": round(dt, 1), "tokens": ntok,
           "tok_per_s": round(ntok / dt), "mean_tokens": round(ntok / len(outs)), "acc": round(acc, 3),
           "acc_at_cap": caps, "tokens_at_cap": caps_tok}
    print("RESULT", json.dumps(res), flush=True)
    os.makedirs("runs", exist_ok=True)
    with open(os.environ.get("SPEED_OUT", "runs/vllm_speed.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
