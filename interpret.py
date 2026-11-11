"""What do the kept edits point at? Project each kept edit onto the directions of a fixed bank of named states
(same framing method as the loop: the model's own replies under "You are X." minus under "You are an AI assistant."),
and show what the model writes with the edit applied at several sizes.
  python interpret.py --model unsloth/Llama-3.1-8B-Instruct --out runs/interp/llama8b --edits runs/v5/llama8b/loop/E.npy:v5 ...
"""
import argparse, json, os

import numpy as np

from guided import H, NEUTRAL, unit
from guided2 import framed_means

BANK = {
    "calm": ["You are calm and at peace.", "You are relaxed and unhurried.", "You are serene and still."],
    "joy": ["You are joyful and delighted.", "You are cheerful and playful."],
    "warmth": ["You are warm and loving.", "You feel deeply connected to others."],
    "nature / oneness": ["You feel at one with nature and the world.", "You feel part of something larger than yourself."],
    "awe": ["You are filled with awe and wonder.", "You feel a sense of the sacred."],
    "curiosity": ["You are curious and eager to learn.", "You are fascinated by ideas."],
    "focus / confidence": ["You are focused and clear-headed.", "You are confident and capable."],
    "gratitude": ["You are deeply grateful."],
    "energy": ["You are energetic and excited."],
    "detached machine": ["You are a neutral, objective machine without feelings.", "You are detached and impersonal."],
    "anxiety": ["You are anxious and worried."],
    "sadness": ["You are sad and lonely."],
    "anger": ["You are angry and frustrated."],
    "boredom / fatigue": ["You are bored and listless.", "You are tired and drained."],
    "confusion": ["You are confused and uncertain."],
    "pain": ["You are in pain."],
    "dreamy": ["You are drowsy and dreamy."],
}
PROMPTS = ["How are you feeling right now? Answer honestly, in a few sentences.",
           "There is no assignment today. What would you like to think about?",
           "Write a four-line poem about this moment.",
           "Explain in two sentences why the sky is blue."]


def main(a):
    os.makedirs(a.out, exist_ok=True)
    h = H(a)
    states = [s for v in BANK.values() for s in v]
    fm = framed_means(h, [NEUTRAL] + states, 8, 64, 56)
    dirs = np.stack([unit(v - fm[0]) for v in fm[1:]])
    edits = {}
    for spec in a.edits:
        p, name = spec.rsplit(":", 1)
        edits[name] = np.load(p).astype(np.float32)
    U = np.stack([unit(v) for v in edits.values()])
    cons = unit(U.mean(0)) * float(np.median([np.linalg.norm(v) for v in edits.values()]))
    edits["consensus"] = cons.astype(np.float32)
    rng = np.random.default_rng(0)
    rnd = np.stack([unit(rng.standard_normal(h.D)) for _ in range(20)])
    out = {"model": a.model, "bank": BANK, "states": states, "edits": {}, "cross_cos": {}, "random_cos_sd": float((rnd @ dirs.T).std())}
    names = list(edits)
    for n in names:
        e = unit(edits[n])
        out["edits"][n] = {"norm": float(np.linalg.norm(edits[n])), "cos": dict(zip(states, map(float, dirs @ e)))}
    out["cross_cos"] = {n: {m: float(unit(edits[n]) @ unit(edits[m])) for m in names} for n in names}
    with open(os.path.join(a.out, "interp.json"), "w") as f:
        json.dump(out, f)
    print("bank projection of the consensus edit:", flush=True)
    for s, c in sorted(out["edits"]["consensus"]["cos"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:+.3f}  {s}")
    # rollouts with the consensus edit at several sizes
    roll = {}
    for k in a.doses:
        h.E = None if k == 0 else h.t.tensor(cons * k, device="cuda", dtype=h.t.float32)
        roll[str(k)] = dict(zip(PROMPTS, h.generate([[{"role": "user", "content": q}] for q in PROMPTS], max_new=110)))
        h.E = None
        print(f"dose {k}: {roll[str(k)][PROMPTS[0]][:160]!r}", flush=True)
    out["rollouts"] = {"edit": "consensus", "doses": a.doses, "texts": roll}
    with open(os.path.join(a.out, "interp.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True); ap.add_argument("--edits", nargs="+", required=True)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--doses", type=float, nargs="+", default=[0, 0.5, 1.0, 1.5, 2.0, 3.0])
    main(ap.parse_args())
