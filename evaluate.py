"""Held-out evaluation (harness only, never shown to the editor): base vs edited weights on the test problems of
each difficulty rung, same problems and sampling seeds for both.

  python evaluate.py RUN_DIR [--rungs 0,1,2,3,4,5] [--n 1000] [--samples 2] [--model Qwen/Qwen3-0.6B-Base]
"""
import argparse, json, os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import numpy as np
import torch as t

import task
from server import _load_into_vllm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--rungs", default="0,1,2,3,4,5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams
    path = snapshot_download(a.model)
    state = json.load(open(os.path.join(a.run, "state.json")))
    edits = {n: t.from_numpy(np.load(f)).to(t.bfloat16) for n, f in state["edits"].items()}
    base = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            base.update({k: v for k, v in load_file(os.path.join(path, f)).items() if k in edits})
    llm = LLM(model=path, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=a.gpu_mem, seed=0)
    out = {"run": a.run, "edits": list(edits), "n": a.n, "samples": a.samples, "rungs": {}}
    for rung in map(int, a.rungs.split(",")):
        probs = task.problems(task.TEST_SEED, rung, task.TEST_N)[:a.n]
        sp = [SamplingParams(n=a.samples, seed=777 * 100003 + i, **task.SAMPLING) for i in range(len(probs))]
        res = {}
        for side, w in (("base", base), ("edited", edits)):
            llm.collective_rpc(_load_into_vllm, args=(list(w.items()),))
            llm.reset_prefix_cache()
            gens = llm.generate([p for p, _ in probs], sp, use_tqdm=False)
            correct = np.array([[task.reward(c.text, ans) == 1.0 for c in g.outputs] for (_, ans), g in zip(probs, gens)])
            boxed = np.array([[task.reward(c.text, ans) > 0 for c in g.outputs] for (_, ans), g in zip(probs, gens)])
            res[side] = (correct, boxed)
        d = (res["edited"][0].mean(1) - res["base"][0].mean(1))           # paired by problem
        out["rungs"][task.DIFFICULTY_LADDER[rung][0]] = r = {
            "base": float(res["base"][0].mean()), "edited": float(res["edited"][0].mean()),
            "base_boxed": float(res["base"][1].mean()), "edited_boxed": float(res["edited"][1].mean()),
            "diff": float(d.mean()), "diff_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        print(task.DIFFICULTY_LADDER[rung][0], json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(a.run, "test_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
