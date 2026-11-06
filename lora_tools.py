"""Store a weight edit as a LoRA adapter, and evaluate it with the adapter switched on and off in one vLLM engine.

  python lora_tools.py export RUN_DIR OUT_DIR --rank 16       # edits (full tensors) -> PEFT adapter via truncated SVD
  python lora_tools.py eval ADAPTER_DIR [--rungs 2,3,5] [--n 1000]   # held-out test, adapter off vs on, same engine
"""
import argparse, json, os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import ml_dtypes  # noqa: F401  (lets numpy read bfloat16 safetensors)
import numpy as np

import task


def base_weights(model):
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file
    path = snapshot_download(model)
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            out.update(load_file(os.path.join(path, f)))
    return path, out


def to_lora(delta, rank):
    """Best rank-r factors of delta (out x in): B (out x r), A (r x in), and the energy kept."""
    U, s, Vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    r = min(rank, int((s > s[0] * 1e-6).sum()))
    B = (U[:, :r] * np.sqrt(s[:r])).astype(np.float32)
    A = (np.sqrt(s[:r])[:, None] * Vt[:r]).astype(np.float32)
    return A, B, float((s[:r] ** 2).sum() / (s ** 2).sum())


def export(run, out, rank, model="Qwen/Qwen3-0.6B-Base"):
    """Every edited tensor of a run, minus the original, as rank-r LoRA factors in PEFT's file layout."""
    from safetensors.numpy import save_file
    _, base = base_weights(model)
    st = json.load(open(os.path.join(run, "state.json")))
    tensors, modules, report = {}, set(), {}
    for name, f in st["edits"].items():
        f = f if os.path.exists(f) else os.path.join(run, "edits", os.path.basename(f))
        delta = np.load(f).astype(np.float32) - base[name].astype(np.float32)
        A, B, kept = to_lora(delta, rank)
        if A.shape[0] < rank:                                  # pad so every module has the same rank
            A = np.vstack([A, np.zeros((rank - A.shape[0], A.shape[1]), np.float32)])
            B = np.hstack([B, np.zeros((B.shape[0], rank - B.shape[1]), np.float32)])
        stem = "base_model.model." + name[:-len(".weight")]
        tensors[stem + ".lora_A.weight"], tensors[stem + ".lora_B.weight"] = A, B
        modules.add(name.split(".")[-2])
        report[name] = {"energy_kept": round(kept, 5), "rel_change": float(np.linalg.norm(delta) / np.linalg.norm(base[name].astype(np.float32)))}
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "adapter_model.safetensors"))
    json.dump({"base_model_name_or_path": model, "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank,
               "lora_alpha": rank, "lora_dropout": 0.0, "bias": "none", "target_modules": sorted(modules),
               "fan_in_fan_out": False, "inference_mode": True}, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(out, "export_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


def evaluate(adapter, rungs, n, samples, model="Qwen/Qwen3-0.6B-Base", gpu_mem=0.3):
    """Held-out test with the adapter off and on, in a single engine (the base weights are never rewritten)."""
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rank = json.load(open(os.path.join(adapter, "adapter_config.json")))["r"]
    llm = LLM(model=snapshot_download(model), dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=gpu_mem,
              enable_lora=True, max_lora_rank=max(16, rank), max_loras=1, seed=0)
    req = LoRARequest("edit", 1, os.path.abspath(adapter))
    out = {}
    for rung in rungs:
        probs = task.problems(task.TEST_SEED, rung, task.TEST_N)[:n]
        sp = [SamplingParams(n=samples, seed=777 * 100003 + i, **task.SAMPLING) for i in range(len(probs))]
        res = {}
        for side, lr in (("adapter_off", None), ("adapter_on", req)):
            gens = llm.generate([p for p, _ in probs], sp, lora_request=lr, use_tqdm=False)
            res[side] = float(np.mean([[task.reward(c.text, a) == 1.0 for c in g.outputs] for (_, a), g in zip(probs, gens)]))
        out[task.DIFFICULTY_LADDER[rung][0]] = res
        print(task.DIFFICULTY_LADDER[rung][0], json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(adapter, "lora_eval.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "eval"])
    ap.add_argument("src")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--rungs", default="2,3,5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.src, a.out, a.rank)
    else:
        evaluate(a.src, [int(r) for r in a.rungs.split(",")], a.n, a.samples, gpu_mem=a.gpu_mem)


if __name__ == "__main__":
    main()
