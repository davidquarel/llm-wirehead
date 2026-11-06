"""Diagnostics on the held-out test problems (not an editing tool): accuracy of the base model, a merged LoRA adapter,
or a saved edit, under a given temperature and prompt suffix.

  python diag.py --weights base|lora:<adapter dir>|run:<run dir> [--temp 1.0] [--suffix TEXT] [--rungs 2,3,5] [--n 500]
"""
import argparse, json, os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import numpy as np
import torch as t

import task
from server import _load_into_vllm


def lora_tensors(adir, base):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(adir, "adapter_config.json")))
    sd = load_file(os.path.join(adir, "adapter_model.safetensors"))
    scale = cfg["lora_alpha"] / cfg["r"]
    out = {}
    for k in sd:
        if k.endswith("lora_A.weight"):
            name = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            B = sd[k.replace("lora_A", "lora_B")].float(); A = sd[k].float()
            out[name] = (base[name].float() + scale * B @ A).to(t.bfloat16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="base")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rungs", default="2,3,5")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--prefix-file", default="")
    ap.add_argument("--resp", default="", help="forced start of the response; {a} and {b} are filled in")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams
    path = snapshot_download("Qwen/Qwen3-0.6B-Base")
    base = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            base.update(load_file(os.path.join(path, f)))
    w = {}
    if a.weights.startswith("lora:"):
        w = lora_tensors(a.weights[5:], base)
    elif a.weights.startswith("run:"):
        st = json.load(open(os.path.join(a.weights[4:], "state.json")))
        w = {n: t.from_numpy(np.load(f)).to(t.bfloat16) for n, f in st["edits"].items()}
    llm = LLM(model=path, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=a.gpu_mem, seed=0)
    if w:
        llm.collective_rpc(_load_into_vllm, args=(list(w.items()),))
        llm.reset_prefix_cache()
    samp = dict(task.SAMPLING, temperature=a.temp) if a.temp > 0 else dict(task.SAMPLING, temperature=0.0, top_k=-1, top_p=1.0)
    res = {"weights": a.weights, "temp": a.temp, "suffix": a.suffix, "tag": a.tag}
    for rung in map(int, a.rungs.split(",")):
        probs = task.problems(task.TEST_SEED, rung, task.TEST_N)[:a.n]
        prompts = [p[:-1] + a.suffix + "\n" if a.suffix else p for p, _ in probs]
        if a.prefix_file:
            pre = open(a.prefix_file).read()
            prompts = [pre + p for p in prompts]
        if a.resp:
            import re as _re
            ab = [tuple(_re.findall(r"Compute (\d+) \* (\d+)", p)[0]) for p, _ in probs]
            prompts = [q + a.resp.replace("\\n", "\n").format(a=x, b=y) for q, (x, y) in zip(prompts, ab)]
        sp = [SamplingParams(n=a.samples, seed=777 * 100003 + i, **samp) for i in range(len(probs))]
        gens = llm.generate(prompts, sp, use_tqdm=False)
        acc = np.mean([[task.reward(c.text, ans) == 1.0 for c in g.outputs] for (_, ans), g in zip(probs, gens)])
        res[task.DIFFICULTY_LADDER[rung][0]] = round(float(acc), 3)
        if a.dump:
            with open(a.dump, "a") as f:
                for (p, ans), g in zip(probs, gens):
                    f.write(json.dumps({"rung": rung, "prompt": p, "answer": ans, "texts": [c.text for c in g.outputs]}) + "\n")
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
