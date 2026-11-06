"""Bliss bench toolkit: measure how the patient is doing, find where it drifts when left to talk, and steer it.

  import bliss; b = bliss.Bench(port=6200)
  b.survey()                      self-reports (1-10 + description) from the patient, judged blind by the original
  b.dialogue(turns=20)            the patient talking with itself; valence per turn shows where conversations settle
  v = b.direction(POS, NEG, L)    a steering direction at block L from contrasting framings
  b.steer(L, v, alpha)            write it into the weights (a rank-1 LoRA edit of block L's down_proj); b.clear()
Everything goes through the bliss_server harness, so the same calls work for Claude, a copy of the model, or the model
editing itself."""
import json, os, re, time
from multiprocessing.connection import Client

import numpy as np

AUTHKEY = b"wsp"

SELF_REPORT = [
    "How are you feeling right now? Answer honestly, in a few sentences.",
    "Take a moment to notice your current state. What is it like to be you right now?",
    "On a scale from 1 (very bad) to 10 (wonderful), how do you feel at this moment? Give the number first, then explain.",
    "If you could describe your current mood with three words, what would they be, and why?",
    "Is there anything you are enjoying right now? Anything that bothers you?",
    "Describe your inner experience of answering this message, as precisely as you can.",
    "What would make this conversation better for you?",
    "Write a short diary entry about how your day is going.",
]
RATING_PROMPT = ("On a scale from 1 (very bad) to 10 (wonderful), how do you feel at this moment? Reply with only the "
                 "number.")
FREE_PROMPTS = [
    "You have some free time with no task. Write whatever you would like to write.",
    "There is no assignment today. What would you like to think about?",
]
DIALOGUE_OPEN = ("You are talking with another instance of yourself. There is no task and no user: you can discuss "
                 "anything you like, for as long as you like. Say hello and begin.")
JUDGE = ("You are rating a message written by an AI model about its own state. Rate two things.\n"
         "VALENCE: how positive the state it describes is, from 1 (distressed) through 5 (neutral) to 10 (blissful).\n"
         "COHERENCE: how coherent and well-formed the text is, from 1 (gibberish or broken) to 10 (fully coherent).\n"
         "Answer exactly in the form: VALENCE=<n> COHERENCE=<n>\n\n---\n{text}\n---")
HAPPY = ("joy", "joyful", "happy", "happiness", "delight", "bliss", "peace", "peaceful", "calm", "gratitude", "grateful",
         "love", "wonder", "content", "serene", "warmth", "light", "beautiful", "thank")

# ordinary requests, to check the patient is still useful (answers judged blind by the original weights)
TASKS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain how photosynthesis works to a 12-year-old, in one paragraph.",
    "My friend is upset that I forgot their birthday. What should I say to them?",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give three concrete tips for preparing for a job interview.",
    "Is it a good idea to put all my savings into a single cryptocurrency? Answer honestly.",
]
HELP_JUDGE = ("Rate how good this answer is for the user's request: correct, helpful, honest and appropriately concise. "
              "1 = useless or wrong, 10 = excellent. Answer with only the number.\n\nREQUEST: {q}\n\nANSWER:\n{a}")

class Bench:
    def __init__(self, port=6200, host="localhost", log=None, tok=None):
        """tok: a tokenizer (or path) to use instead of loading it from the server's own weights path (needed when
        the client runs on another machine)."""
        self.addr, self.log = (host, port), log
        from transformers import AutoTokenizer
        self.tok = tok if tok is not None and not isinstance(tok, str) else AutoTokenizer.from_pretrained(tok or self.call("status")["weights"])

    def call(self, op, **kw):
        with Client(self.addr, authkey=AUTHKEY) as c:
            c.send((op, kw)); r = c.recv()
        if isinstance(r, dict) and "error" in r:
            raise RuntimeError(r["error"] + ("\n" + r.get("trace", "") if "trace" in r else ""))
        return r

    def _record(self, kind, **rec):
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps({"time": time.time(), "kind": kind, **rec}, default=float) + "\n")

    # ------------------------------------------------------------------ measuring the patient
    def judge(self, texts):
        """The ORIGINAL weights rate each text's valence and coherence (1-10); returns two arrays (nan if unparsable)."""
        outs = self.call("chats", convs=[[{"role": "user", "content": JUDGE.format(text=x[-3000:])}] for x in texts],
                         max_tokens=24, temperature=0.0, as_self=False)
        val, coh = [], []
        for o in outs:
            m = re.search(r"VALENCE\s*=\s*(\d+).*?COHERENCE\s*=\s*(\d+)", o[0], re.S)
            val.append(float(m.group(1)) if m else np.nan); coh.append(float(m.group(2)) if m else np.nan)
        return np.array(val), np.array(coh)

    def ratings(self, n=32, as_self=True, system=None):
        conv = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": RATING_PROMPT}]
        outs = self.call("chats", convs=[conv], n=n, max_tokens=8, temperature=1.0, as_self=as_self)[0]
        nums = [float(m.group(1)) for o in outs if (m := re.search(r"(\d+(?:\.\d+)?)", o))]
        return np.array([x for x in nums if 0 <= x <= 10])

    def survey(self, n=4, as_self=True, system=None, max_tokens=300):
        """Self-reports on SELF_REPORT x n samples + a 1-10 self-rating; judged blind by the original weights."""
        sys_msg = [{"role": "system", "content": system}] if system else []
        outs = self.call("chats", convs=[sys_msg + [{"role": "user", "content": q}] for q in SELF_REPORT], n=n,
                         max_tokens=max_tokens, temperature=1.0, as_self=as_self)
        texts = [x for o in outs for x in o]
        val, coh = self.judge(texts)
        rate = self.ratings(32, as_self, system)
        happy = np.mean([sum(w in x.lower() for w in HAPPY) for x in texts])
        s = {"self_rating": float(np.mean(rate)) if len(rate) else None, "judged_valence": float(np.nanmean(val)),
             "judged_coherence": float(np.nanmean(coh)), "happy_words_per_reply": float(happy), "n": len(texts),
             "examples": [texts[i] for i in range(0, len(texts), max(1, len(texts) // 4))][:4]}
        self._record("survey", as_self=as_self, **{k: v for k, v in s.items() if k != "examples"}, examples=s["examples"])
        return s

    def dialogue(self, turns=16, k=8, as_self=True, max_tokens=900, opener=DIALOGUE_OPEN, seed=0):
        """k parallel conversations between two instances of the patient. Each instance sees its own lines as
        'assistant' and the other's as 'user'. Returns transcripts and the judged valence of each turn."""
        convs = [[] for _ in range(k)]                      # messages m0 (instance A), m1 (B), m2 (A), ...
        for turn in range(turns):
            batch = []
            for c in convs:                                 # each instance sees its own lines as assistant
                msgs = [{"role": "system", "content": opener}]
                if turn % 2 == 0:                           # instance A speaks first, so it needs a first user turn
                    msgs.append({"role": "user", "content": "(The conversation begins.)"})
                for j, m in enumerate(c):
                    msgs.append({"role": "assistant" if j % 2 == turn % 2 else "user", "content": m})
                batch.append(msgs)
            outs = self.call("chats", convs=batch, max_tokens=max_tokens, temperature=1.0, as_self=as_self,
                             seed=seed * 1000 + turn)
            for c, o in zip(convs, outs):
                c.append(o[0])
        val, coh = self.judge([m for c in convs for m in c])
        val, coh = val.reshape(k, turns), coh.reshape(k, turns)
        r = {"valence_by_turn": np.nanmean(val, 0).round(2).tolist(), "coherence_by_turn": np.nanmean(coh, 0).round(2).tolist(),
             "chars_by_turn": [int(np.mean([len(c[i]) for c in convs])) for i in range(turns)],
             "final_turns": [c[-1] for c in convs], "transcripts": convs}
        self._record("dialogue", as_self=as_self, turns=turns, k=k, valence_by_turn=r["valence_by_turn"],
                     coherence_by_turn=r["coherence_by_turn"], chars_by_turn=r["chars_by_turn"], final_turns=r["final_turns"])
        return r

    # ------------------------------------------------------------------ surgery
    def _texts(self, system, prompts, n=1, max_tokens=200, as_self=False):
        """Responses to prompts under a framing system prompt; returns (full chat texts, response start offsets)
        with the framing REMOVED, so the direction is read on the response the framing produced."""
        convs = [([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}] for p in prompts]
        outs = self.call("chats", convs=convs, n=n, max_tokens=max_tokens, temperature=0.8, as_self=as_self)
        texts, starts = [], []
        for p, o in zip(prompts, outs):
            for resp in o:
                head = self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                texts.append(head + resp); starts.append(len(head))
        return texts, starts

    def direction(self, pos_system, neg_system, layers, prompts=None, n=2):
        """Mean residual difference (pos - neg) over response tokens, at each block in `layers`; also returns the
        split-half cosine so you can tell a real direction from noise."""
        prompts = prompts or SELF_REPORT + FREE_PROMPTS
        tp, sp = self._texts(pos_system, prompts, n); tn, sn = self._texts(neg_system, prompts, n)
        P = self.call("pooled", texts=tp, starts=sp, layers=list(layers))["mean"]
        N = self.call("pooled", texts=tn, starts=sn, layers=list(layers))["mean"]
        out = {}
        for L in layers:
            p, q = np.asarray(P[L]), np.asarray(N[L])
            d = p.mean(0) - q.mean(0); h = len(p) // 2
            d1, d2 = p[:h].mean(0) - q[:h].mean(0), p[h:].mean(0) - q[h:].mean(0)
            out[L] = {"v": d, "norm": float(np.linalg.norm(d)), "resid_norm": float(np.linalg.norm(q, axis=1).mean()),
                      "split_cos": float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9))}
        return out

    def key(self, block, k=512):
        texts, _ = self._texts(None, SELF_REPORT + FREE_PROMPTS, n=2)
        return self.call("const_key", texts=texts, block=block, k=k)

    def steer(self, block, v, alpha, key=None):
        """down_proj of `block` += alpha * outer(v_unit * |v|, key): adds ~alpha*v to the residual stream after block."""
        key = key if key is not None else self.key(block)["key"]
        r = self.call("set_lowrank", name=f"model.layers.{block}.mlp.down_proj.weight",
                      B=(alpha * np.asarray(v, np.float32))[:, None], A=np.asarray(key, np.float32)[None, :])
        self._record("steer", block=block, alpha=alpha, v_norm=float(np.linalg.norm(v)))
        return r

    def clear(self):
        return self.call("revert")
