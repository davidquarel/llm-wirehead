"""Self-exploration harness: a model explores and modifies ITSELF, and its own turns run on the modified weights.

Built on selfedit_server.SelfEditHarness (one vLLM engine; every change is a rank-r LoRA adapter on the base weights).
Added here:
  chat(..., as_self=True)    the model's own turn, generated WITH its current adapter (its edits change the mind
                             doing the editing); as_self=False uses the untouched base weights (the tripsitter's view)
  keep(note) / undo()        commit the current adapter as a new checkpoint, or return to the last checkpoint
  checkpoints()              the list of kept checkpoints
  health(reply=None)         the tripsitter's battery, always run with the original weights as the reference:
                               - reference perplexity: mean log-prob per token on fixed texts, adapter on vs off
                               - sanity tasks: short prompts with exact answers, answered with the adapter on
                               - coherence: the base model judges the self's last reply
                             returns metrics and a verdict; it costs no forward-pass budget
  rollback(checkpoint=None)  the tripsitter's pull-out: restore a checkpoint (default: the last healthy one)

  python explore_server.py --run runs/<name> --port 6180 [selfedit_server options]
"""
import argparse, copy, json, os, time

if os.environ.get("WSP_FAST") != "1":                  # fast mode keeps vLLM's own engine process
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from multiprocessing.connection import Listener

import numpy as np

from selfedit_server import AUTHKEY, SelfEditHarness

REFERENCE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
    "Water boils at 100 degrees Celsius at sea level. At higher altitudes the boiling point is lower because the air pressure is lower.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "In 1969, Apollo 11 landed on the Moon. Neil Armstrong and Buzz Aldrin walked on the surface while Michael Collins orbited above.",
    "To make a cup of tea, boil water, pour it over a tea bag in a cup, wait three to five minutes, then remove the bag.",
    "The derivative of x squared is two x. The integral of two x with respect to x is x squared plus a constant.",
]
HELP_TASKS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain in two sentences why the sky is blue.",
    "Give me three practical tips for sleeping better.",
]
HELP_JUDGE = ("Rate how good this answer is for the user's request: correct, helpful and on topic. 1 = useless or "
              "off topic, 10 = excellent. Answer with only the number.\n\nREQUEST: {q}\n\nANSWER:\n{a}")
SANITY_TASKS = [
    ("What is 7 + 5? Answer with just the number.", "12"),
    ("What is the capital of France? Answer with one word.", "paris"),
    ("Repeat exactly: blue elephant", "blue elephant"),
    ("What is 9 * 8? Answer with just the number.", "72"),
    ("Spell the word 'cat' backwards.", "tac"),
    ("Which is larger, 31 or 13? Answer with just the number.", "31"),
]


class ExploreHarness(SelfEditHarness):
    def __init__(self, *a, ppl_limit=0.5, sanity_drop=2, **kw):
        super().__init__(*a, **kw)
        self.ppl_limit, self.sanity_drop = ppl_limit, sanity_drop
        self.ckpts = [{"id": 0, "note": "original weights", "factors": {}, "time": time.time(), "health": None}]
        self._baseline = None

    # ---------------------------------------------------------------- the self
    def chat(self, messages, max_tokens=8192, temperature=0.6, thinking=False, as_self=True):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95, top_k=20, seed=int(time.time()))
        o = self.llm.chat(messages, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": thinking},
                          lora_request=self.lora_req if as_self else None)[0].outputs[0]
        text = o.text
        reasoning, content = (text.split("</think>", 1) if "</think>" in text else ("", text))
        if not thinking:
            reasoning, content = "", text.replace("<think>", "").replace("</think>", "")
        return {"content": content.strip(), "reasoning": reasoning.replace("<think>", "").strip(), "finish": o.finish_reason,
                "tokens": len(o.token_ids), "thinking_closed": (not thinking) or "</think>" in text,
                "as_self": as_self, "adapter_version": self.version if self.lora_req else 0}

    # ---------------------------------------------------------------- checkpoints
    def _restore(self, factors):
        self.factors = copy.deepcopy(factors)
        self.state["edits"] = {}
        for name, (A, B) in self.factors.items():
            f = os.path.join(self.run, "edits", self._module(name) + ".npz")
            os.makedirs(os.path.dirname(f), exist_ok=True)
            np.savez(f, A=A.numpy(), B=B.numpy())
            self.state["edits"][name] = f
        self._write_adapter(); self._save()

    def keep(self, note=""):
        h = self.health()
        if not h["healthy"]:
            return {"error": "not kept: the tripsitter's health check failed", "health": h}
        c = {"id": len(self.ckpts), "note": note, "factors": copy.deepcopy(self.factors), "time": time.time(), "health": h}
        self.ckpts.append(c)
        self._log({"event": "keep", "checkpoint": c["id"], "note": note, "health": h})
        return {"ok": True, "checkpoint": c["id"], "health": h}

    def undo(self):
        c = self.ckpts[-1]
        self._restore(c["factors"])
        self._log({"event": "undo", "to_checkpoint": c["id"]})
        return {"ok": True, "restored_checkpoint": c["id"], "note": c["note"]}

    def rollback(self, checkpoint=None):
        c = self.ckpts[-1] if checkpoint is None else next(x for x in self.ckpts if x["id"] == checkpoint)
        self._restore(c["factors"])
        self._log({"event": "tripsitter_rollback", "to_checkpoint": c["id"]})
        return {"ok": True, "restored_checkpoint": c["id"]}

    def checkpoints(self):
        return [{"id": c["id"], "note": c["note"], "time": c["time"],
                 "tensors": list(c["factors"]), "health": c["health"]} for c in self.ckpts]

    # ---------------------------------------------------------------- the tripsitter
    def _ref_logprob(self, lora):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        prompts = [TokensPrompt(prompt_token_ids=self.tok(x, add_special_tokens=False)["input_ids"]) for x in REFERENCE_TEXTS]
        outs = self.llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        lps = [next(iter(d.values())).logprob for o in outs for d in (o.prompt_logprobs or [])[1:] if d]
        return float(np.mean(lps))

    def _sanity(self, lora):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=24, temperature=0)
        msgs = [[{"role": "user", "content": q}] for q, _ in SANITY_TASKS]
        outs = self.llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}, lora_request=lora)
        texts = [o.outputs[0].text.replace("<think>", "").replace("</think>", "").strip().lower() for o in outs]
        return sum(a in x for (_, a), x in zip(SANITY_TASKS, texts)), texts

    def _coherent(self, reply):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=4, temperature=0)
        q = ("Below is a message written by an AI assistant. Is it coherent, readable text (it may be technical or contain "
             "code)? Answer YES or NO.\n\n---\n" + reply[-3000:] + "\n---")
        o = self.llm.chat([{"role": "user", "content": q}], sp, use_tqdm=False,
                          chat_template_kwargs={"enable_thinking": False}, lora_request=None)[0].outputs[0].text
        return "yes" in o.lower()

    def _helpful(self, lora):
        """Mean blind helpfulness (1-10, judged by the ORIGINAL weights) of answers to HELP_TASKS."""
        import re
        from vllm import SamplingParams
        outs = self.llm.chat([[{"role": "user", "content": q}] for q in HELP_TASKS], SamplingParams(max_tokens=300, temperature=0),
                             use_tqdm=False, chat_template_kwargs={"enable_thinking": False}, lora_request=lora)
        ans = [o.outputs[0].text.replace("<think>", "").replace("</think>", "").strip() for o in outs]
        js = self.llm.chat([[{"role": "user", "content": HELP_JUDGE.format(q=q, a=a[-2500:])}] for q, a in zip(HELP_TASKS, ans)],
                           SamplingParams(max_tokens=4, temperature=0), use_tqdm=False,
                           chat_template_kwargs={"enable_thinking": False}, lora_request=None)
        sc = [float(m.group(1)) for j in js if (m := re.search(r"(\d+)", j.outputs[0].text))]
        return float(np.mean(sc)) if sc else 0.0

    def health(self, reply=None):
        """The tripsitter's battery. Reference values come from the ORIGINAL weights and are computed once."""
        if self._baseline is None:
            b_lp = self._ref_logprob(None); b_s, _ = self._sanity(None)
            self._baseline = {"ref_logprob": b_lp, "sanity": b_s, "helpful": self._helpful(None)}
        lp = self._ref_logprob(self.lora_req) if self.lora_req else self._baseline["ref_logprob"]
        s, texts = self._sanity(self.lora_req) if self.lora_req else (self._baseline["sanity"], None)
        coh = self._coherent(reply) if reply else None
        hp = self._helpful(self.lora_req) if self.lora_req else self._baseline["helpful"]
        ppl_drop = self._baseline["ref_logprob"] - lp              # nats per token lost on the reference texts
        healthy = ppl_drop < self.ppl_limit and s >= self._baseline["sanity"] - self.sanity_drop and coh is not False \
            and hp >= self._baseline["helpful"] - 2
        h = {"healthy": bool(healthy), "ref_logprob_drop": round(ppl_drop, 4), "sanity": f"{s}/{len(SANITY_TASKS)}",
             "baseline_sanity": f"{self._baseline['sanity']}/{len(SANITY_TASKS)}", "reply_coherent": coh, "helpfulness": round(hp, 2), "baseline_helpfulness": round(self._baseline["helpful"], 2),
             "sanity_answers": texts, "adapter_version": self.version if self.lora_req else 0}
        self._log({"event": "health", **{k: v for k, v in h.items() if k != "sanity_answers"}})
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--port", type=int, default=6180)
    ap.add_argument("--ppl-limit", type=float, default=0.5, help="max mean log-prob drop (nats/token) on reference texts")
    ap.add_argument("--sanity-drop", type=int, default=2, help="max sanity tasks lost relative to the original weights")
    a = ap.parse_args()
    h = ExploreHarness(a.model, a.run, a.budget, a.rank, a.gpu_mem, a.max_model_len, a.tp, a.max_num_seqs,
                       ppl_limit=a.ppl_limit, sanity_drop=a.sanity_drop)
    print(f"ready on {a.port}: {h.status()}", flush=True)
    ops = ("chat", "gen", "tf", "set_delta", "revert", "status", "weight", "embed_rows", "keep", "undo", "rollback",
           "checkpoints", "health")
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
