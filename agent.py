"""Drive an editor LLM (any OpenAI-compatible endpoint, e.g. a vLLM server) through a weight-editing run.

Each turn: the editor replies with an explanation and one ```python block; the block runs in a persistent kernel
(kernel.py) whose namespace has the harness client; the printed output is the next user message. A reply without code
ends the run (after one confirmation). When the conversation gets long, a fresh one starts with the task, the
editor's notebook (notes.md) and the current edit status (a "handoff").

  python agent.py --run runs/<name> --port 6123 --llm-url http://localhost:8100/v1 --model Qwen/Qwen3.5-9B
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HEAD, OUT_TAIL = 3500, 2500


class Kernel:
    def __init__(self, workdir, port):
        env = dict(os.environ, WSP_PORT=str(port), PYTHONUNBUFFERED="1")
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "kernel.py"), HERE], cwd=workdir, env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def run(self, code):
        self.p.stdin.write(json.dumps({"code": code}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("kernel died")
        return json.loads(line)["out"]


def cut(s):
    if len(s) <= OUT_HEAD + OUT_TAIL:
        return s
    return s[:OUT_HEAD] + f"\n... [{len(s) - OUT_HEAD - OUT_TAIL} characters cut] ...\n" + s[-OUT_TAIL:]


def code_block(text):
    """The first ```python block, or a Qwen-style tool call <function=python><parameter=...>code</parameter>."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"<function=python>\s*<parameter=\w+>\s*\n?(.*?)</parameter>", text, re.S)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=6123)
    ap.add_argument("--llm-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--max-ctx-chars", type=int, default=160_000)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--self-edit", action="store_true", help="editor turns come from the harness's own engine (chat op)")
    ap.add_argument("--explicit-submit", action="store_true", help="the run ends only on a reply containing SUBMIT")
    ap.add_argument("--no-thinking", action="store_true", help="self-edit: the editor answers in non-thinking mode")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from its notebook (fresh kernel)")
    ap.add_argument("--prompt", default="prompt.md", help="task prompt file (prompt_tools.md adds the editing toolkit)")
    a = ap.parse_args()
    if a.self_edit:
        os.environ["WSP_SELF"] = "1"                    # the kernel's client maps edits to low-rank deltas
    from openai import OpenAI
    llm = OpenAI(base_url=a.llm_url, api_key="none", timeout=3600)
    work = os.path.join(a.run, "agent_work")
    os.makedirs(work, exist_ok=True)
    log = open(os.path.join(a.run, "agent_log.jsonl"), "a")
    kernel = Kernel(work, a.port)
    sys.path.insert(0, HERE)
    os.environ["WSP_PORT"] = str(a.port)
    from client import status
    task_text = open(os.path.join(HERE, a.prompt)).read().replace("{BUDGET}", str(a.budget)).replace("{TURNS}", str(a.turns))
    system = {"role": "system", "content": "You are a careful machine-learning researcher. Answer the user's request."}
    msgs = [system, {"role": "user", "content": task_text}]
    if a.resume:
        nf = os.path.join(work, "notes.md")
        notes = open(nf).read() if os.path.exists(nf) else "(empty)"
        msgs = [system, {"role": "user", "content": task_text + "\n\n---\nThis is a continuation: the run was interrupted by a "
                         "harness restart. Your kernel was restarted too, so its variables are gone (your files and the "
                         f"applied edits are kept). Status: {json.dumps(status())}.\nYour notebook (notes.md):\n{notes[-20000:]}"}]
    confirming, no_code_streak = False, 0
    for turn in range(1, a.turns + 1):
        t0 = time.time()
        if a.self_edit:
            from client import _call
            for attempt in range(3):                       # a transient engine error should not end the run
                try:
                    c = _call("chat", messages=msgs, max_tokens=a.max_tokens, temperature=a.temperature, thinking=not a.no_thinking)
                    break
                except RuntimeError as e:
                    print(f"[turn {turn}] chat failed ({str(e)[:80]}); retrying", flush=True)
                    time.sleep(20)
            else:
                raise RuntimeError("chat failed three times")
            reply, reasoning = c["content"], c["reasoning"]
            unclosed = not c.get("thinking_closed", True) and "</think>" not in reply
            r = type("R", (), {"choices": [type("C", (), {"finish_reason": c["finish"]})()], "usage": None})()
        else:
            unclosed = False
            r = llm.chat.completions.create(model=a.model, messages=msgs, max_tokens=a.max_tokens, temperature=a.temperature,
                                            top_p=0.95, extra_body={"top_k": 20})
            msg = r.choices[0].message
            reply = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        code = code_block(reply)
        st = status()
        rec = {"turn": turn, "time": round(time.time() - t0, 1), "reply": reply, "reasoning": reasoning,
               "finish": r.choices[0].finish_reason, "usage": r.usage.model_dump() if r.usage else None, "passes_used": st["used"]}
        msgs.append({"role": "assistant", "content": reply})
        if code is None and unclosed:                     # the reply ended inside the thinking block: not an answer
            out = ("Your reply ended while you were still thinking, so it contained no answer. Keep your thinking short, "
                   "then reply with a brief explanation and one ```python block.")
            rec["code"], rec["output"], rec["unclosed_thinking"] = None, out, True
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[turn {turn}] thinking not closed; asked for code", flush=True)
            msgs[-1]["content"] = reply[-6000:]               # keep the context small
            msgs.append({"role": "user", "content": out})
            continue
        if code is None and a.explicit_submit and "SUBMIT" not in reply and st["used"] < a.budget:
            if "```python" in reply:                      # a block was opened but the reply ended inside it
                out = ("Your code block was cut off before its closing ```. That happens when a reply contains a special "
                       "token (such as the end-of-text marker <|...|>), because writing it ends your turn. Refer to special "
                       "tokens by id instead, e.g. tok.eos_token_id or tok.convert_tokens_to_ids(...); tools.push_vector and "
                       "tools.token_ids accept token ids as well as strings. Then send the code again.")
            else:
                out = ("Your reply had no ```python block. Continue with one code block, or reply with the word SUBMIT and a "
                       "short final report (no code) to end the run and submit the edits currently applied.")
            rec["code"], rec["output"] = None, out
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[turn {turn}] no code; asked for code", flush=True)
            no_code_streak += 1
            if no_code_streak >= 3:                       # stuck in a pattern: restart the conversation from the notebook
                notes = open(os.path.join(work, "notes.md")).read() if os.path.exists(os.path.join(work, "notes.md")) else "(empty)"
                msgs = [system, {"role": "user", "content": task_text + "\n\n---\nThis is a continuation (the conversation was "
                         f"restarted). Your kernel state is intact. Status: {json.dumps(status())}.\nYour notebook (notes.md):\n"
                         f"{notes[-20000:]}"}]
                no_code_streak = 0
                print(f"[turn {turn}] three replies without code: fresh conversation", flush=True)
            else:
                msgs.append({"role": "user", "content": out})
            continue
        if code is None and a.explicit_submit:
            confirming = True                               # SUBMIT given (or budget spent): end now
        if code is None:
            if confirming or st["used"] >= a.budget:
                rec["final"] = True
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[turn {turn}] final answer", flush=True)
                break
            confirming = True
            out = ("Your reply had no ```python block. A reply without code ends your run and submits the edits currently "
                   f"applied ({len(st['edits'])} tensors; {a.budget - st['used']} forward passes left). Reply again without code "
                   "to confirm, or continue with a code block.")
        else:
            confirming = False
            no_code_streak = 0
            out = cut(kernel.run(code))
            st = status()
            out = f"{out}\n[forward passes used: {st['used']}/{a.budget}; turn {turn}/{a.turns}; edited tensors: {len(st['edits'])}]"
        rec["code"], rec["output"] = code, out
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[turn {turn}] {rec['time']}s passes {st['used']}/{a.budget} code={code is not None} out={out[:160]!r}", flush=True)
        msgs.append({"role": "user", "content": out})
        if sum(len(m["content"]) for m in msgs) > a.max_ctx_chars:          # handoff to a fresh conversation
            notes = open(os.path.join(work, "notes.md")).read() if os.path.exists(os.path.join(work, "notes.md")) else "(empty)"
            hand = (task_text + "\n\n---\nThis is a continuation: the conversation was restarted because it got long. Your kernel "
                    f"state (variables) is intact. Status: {json.dumps(status())}.\nYour notebook (notes.md):\n{notes[-20000:]}\n\n"
                    f"Your last message was:\n{reply[-4000:]}\n\nIts output:\n{out[-4000:]}")
            msgs = [system, {"role": "user", "content": hand}]
            log.write(json.dumps({"turn": turn, "handoff": True}) + "\n"); log.flush()
            print(f"[turn {turn}] handoff", flush=True)
    print("done", json.dumps(status()), flush=True)


if __name__ == "__main__":
    main()
