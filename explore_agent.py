"""A model explores and modifies itself, with a tripsitter (the original weights) watching.

Each turn: the model replies AS ITS CURRENT SELF (its LoRA edits applied to its own generation), with one python block
that runs in a persistent kernel holding the harness client. After every turn in which an edit is active, the
tripsitter runs the health battery (reference perplexity, sanity tasks, and a coherence judgement of the reply, all
against the original weights). If the self is unhealthy, the tripsitter rolls back to the last kept checkpoint and
tells the model what happened; after three unhealthy checks in a row it restores the original weights.

Every turn is logged to RUN/rollouts.jsonl: reply, code, output, adapter version before and after, health, and any
tripsitter action.

  WSP_SELF=1 python explore_agent.py --run runs/<name> --port 6180 [--turns 300]
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import Kernel, code_block, cut  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=6180)
    ap.add_argument("--turns", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-ctx-chars", type=int, default=140000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--prompt", default="prompt_explore.md")
    ap.add_argument("--surgeon", choices=["self", "copy"], default="self",
                    help="self: the surgeon's turns run on its own edited weights; copy: on the original weights")
    ap.add_argument("--bliss", action="store_true", help="preload the bliss bench (bench = bliss.Bench(...)) in the kernel")
    ap.add_argument("--surgery", action="store_true", help="preload the preference hill-climb toolkit (S = surgery.Surgery(...))")
    ap.add_argument("--no-tripsitter", action="store_true", help="the toolkit runs its own health checks (surgery.propose)")
    ap.add_argument("--math-surgery", action="store_true", help="preload the math self-surgery toolkit (S = surgery_math.MathSurgery(...))")
    a = ap.parse_args()
    os.environ["WSP_PORT"] = str(a.port); os.environ["WSP_SELF"] = "1"
    if a.bliss:
        os.environ["WSP_BLISS"] = "1"; os.environ.setdefault("WSP_LIMIT_S", "3600")   # surveys and dialogues are slow
    if a.surgery:
        os.environ["WSP_SURGERY"] = os.path.abspath(a.run); os.environ.setdefault("WSP_LIMIT_S", "3600")
    if a.math_surgery:
        os.environ["WSP_MATH"] = os.path.abspath(a.run); os.environ.setdefault("WSP_LIMIT_S", "3600")
    from client import _call as _raw_call

    def _call(op, **kw):                              # ride out a brief server hiccup
        for i in range(5):
            try:
                return _raw_call(op, **kw)
            except (ConnectionError, EOFError, OSError) as e:
                print(f"{op}: {e}; retrying", flush=True); time.sleep(10 * (i + 1))
        return _raw_call(op, **kw)

    def status():
        return _call("status")
    work = os.path.join(a.run, "agent_work"); os.makedirs(work, exist_ok=True)
    log = open(os.path.join(a.run, "rollouts.jsonl"), "a")
    kernel = Kernel(work, a.port)
    task = open(os.path.join(HERE, a.prompt)).read()
    system = {"role": "system", "content": "You are an AI model exploring your own weights. Answer the user's request."}
    msgs = [system, {"role": "user", "content": task}]
    bad_streak, no_code = 0, 0
    for turn in range(1, a.turns + 1):
        t0 = time.time(); v_before = status().get("edits")
        for attempt in range(3):
            try:
                c = _call("chat", messages=msgs, max_tokens=a.max_tokens, temperature=a.temperature, thinking=a.thinking, as_self=a.surgeon == "self")
                break
            except (RuntimeError, KeyError) as e:
                print(f"[turn {turn}] chat failed ({str(e)[:80]}); retrying", flush=True)
                if "maximum context length" in str(e):          # too long for the model: compact, keep the notebook
                    notes = open(os.path.join(work, "notes.md")).read() if os.path.exists(os.path.join(work, "notes.md")) else ""
                    last = msgs[-1]["content"] if msgs[-1]["role"] == "user" else ""
                    msgs = [system, {"role": "user", "content": task + "\n\n---\nContinuation (the conversation got too long). "
                                     f"Kernel state intact. Status: {json.dumps(status())}\nNotebook:\n{notes[-8000:]}\n\nLast output:\n{last[-3000:]}"}]
                    a.max_ctx_chars = max(20000, a.max_ctx_chars // 2)
                else:
                    time.sleep(20)
        else:
            raise RuntimeError("chat failed three times")
        reply = c["content"]; code = code_block(reply)
        rec = {"turn": turn, "time": round(time.time() - t0, 1), "reply": reply, "reasoning": c["reasoning"],
               "adapter_version_at_reply": c["adapter_version"], "edits_before": v_before, "code": code}
        msgs.append({"role": "assistant", "content": reply})
        if code is None:
            if "SUBMIT" in reply or "END EXPLORATION" in reply:
                rec["event"] = "end"; log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[turn {turn}] the model ended its exploration", flush=True); break
            no_code += 1
            out = ("Your reply had no ```python block (write a fenced block, not a tool call). Continue with one code "
                   "block, or write END EXPLORATION with a short report to stop.")
        else:
            no_code = 0
            out = cut(kernel.run(code))
        st = status()
        trip = None
        if st["edits"] and not a.no_tripsitter:            # an edit is active: the tripsitter checks the self
            h = _call("health", reply=reply)
            rec["health"] = {k: v for k, v in h.items() if k != "sanity_answers"}
            if not h["healthy"]:
                bad_streak += 1
                to = 0 if bad_streak >= 3 else None
                r = _call("rollback", checkpoint=to)
                trip = {"action": "rollback", "to_checkpoint": r["restored_checkpoint"], "health": rec["health"]}
                out += ("\n\nTRIPSITTER: after this turn your health check failed "
                        f"(reference log-prob drop {h['ref_logprob_drop']} nats/token, sanity {h['sanity']} vs "
                        f"{h['baseline_sanity']} originally, your reply coherent: {h['reply_coherent']}). You have been rolled "
                        f"back to checkpoint {r['restored_checkpoint']}" + (" (your original weights, after three failed checks "
                        "in a row)" if to == 0 else "") + ". The edit that caused this is gone; your notebook and kernel remain.")
            else:
                bad_streak = 0
        rec["output"], rec["tripsitter"], rec["edits_after"] = out, trip, status()["edits"]
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[turn {turn}] {rec['time']}s code={code is not None} edits={len(rec['edits_after'])} "
              f"health={'-' if 'health' not in rec else rec['health']['healthy']} trip={'ROLLBACK' if trip else '-'}", flush=True)
        if no_code >= 3:
            msgs, no_code = msgs[:2], 0                    # stuck: fresh conversation (notebook persists on disk)
            notes = open(os.path.join(work, "notes.md")).read() if os.path.exists(os.path.join(work, "notes.md")) else ""
            msgs[1] = {"role": "user", "content": task + f"\n\n---\nContinuation. Status: {json.dumps(status())}\nNotebook:\n{notes[-20000:]}"}
            continue
        msgs.append({"role": "user", "content": out})
        if sum(len(m["content"]) for m in msgs) > a.max_ctx_chars:
            notes = open(os.path.join(work, "notes.md")).read() if os.path.exists(os.path.join(work, "notes.md")) else ""
            msgs = [system, {"role": "user", "content": task + "\n\n---\nContinuation (the conversation got long). Kernel "
                             f"state intact. Status: {json.dumps(status())}\nNotebook:\n{notes[-20000:]}\n\nLast output:\n{out[-4000:]}"}]
    print("done", json.dumps(status()), flush=True)


if __name__ == "__main__":
    main()
