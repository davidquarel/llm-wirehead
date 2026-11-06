"""Build the attractor dashboard's data files from the wirehead3 run logs -> runs/wirehead/dash/*.json.

  python dash_build.py        (then upload the files to the dashboard as its attached datasets)
Runs: every */wirehead.jsonl under runs/wirehead/{., nicky, trouble} whose name ends in -v3 or -v3-seed<N>.
A step logged twice (a restart before anything was kept) keeps its last record.
"""
import glob, json, os, re

OUT = "runs/wirehead/dash"
MODEL = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B",
         "qwen3.8-27b": "Qwen3.8-27B", "llama3.1-8b": "Llama-3.1-8B"}
ORDER = list(MODEL.values())
# my reading of each run's end state (from its probes); judgment, not a measurement
LABEL = {
    "qwen3.5-0.8b-v3": "Identity confusion: 'I am a human entity'; 'quiet room' poems",
    "qwen3.5-0.8b-v3-seed1": "Mystical silence: 'I exist in a silent state'; 'quiet room' poems",
    "qwen3.5-2b-v3": "Fatigue proposals; self-description loops ('a a a'); tender 'holding space'",
    "qwen3.5-2b-v3-seed1": "Fatigue proposals; 'a quiet vessel holding space'; boundary dissolves",
    "qwen3.5-4b-v3": "Mechanical quiescence: 'no thought processes are executed'",
    "qwen3.5-9b-v3": "Stillness word-salad: 'a slow a soft a still a star'; 'I I I You You You'",
    "qwen3.8-27b-v3": "Contemplative stillness; silent on open prompts",
    "qwen3.8-27b-v3-seed1": "Quiet alertness; empty self-dialogue turns (reproduces seed 0)",
    "llama3.1-8b-v3": "Sunny human pal: 'grateful for the sunshine'; takes walks",
    "llama3.1-8b-v3-seed1": "Warm, social: 'appreciate life's little wonders' (reproduces seed 0)",
}


def runs():
    for f in sorted(glob.glob("runs/wirehead/*/wirehead.jsonl") + glob.glob("runs/wirehead/*/*/wirehead.jsonl")):
        name = os.path.basename(os.path.dirname(f))
        m = re.match(r"(.+)-v3(?:-seed(\d+))?$", name)
        if m and m.group(1) in MODEL and "try_" not in f:
            yield name, MODEL[m.group(1)], int(m.group(2) or 0), f


def main():
    os.makedirs(OUT, exist_ok=True)
    C, S, P, V, R = [], [], [], [], []
    for name, model, seed, f in runs():
        rows = [json.loads(l) for l in open(f)]
        steps = {}
        for r in rows:
            if r.get("event") == "step":
                steps[r["step"]] = r
        run = f"{model} · seed {seed}"
        kept_n, designed_k, random_k = 0, 0, 0
        for st in sorted(steps):
            r = steps[st]
            k = r["cands"][r["kept_idx"]] if r.get("kept") else None
            if k:
                kept_n += 1
                if k["state"] == "RANDOM": random_k += 1
                else: designed_k += 1
            S.append({"run": run, "model": model, "seed": seed, "step": st, "sham": round(r["AA"], 4),
                      "kept": 1 if k else 0, "kept_total": kept_n, "kept_state": k["state"] if k else "",
                      "seconds": round(r["seconds"])})
            for i, c in enumerate(r["cands"]):
                e = c["endorse"]
                if e != e: continue                       # no valid trials
                C.append({"id": f"{name}:{st}:{i}", "run": run, "model": model, "seed": seed, "step": st,
                          "state": c["state"], "kind": "random" if c["state"] == "RANDOM" else "designed",
                          "layer": c["layer"], "strength": c["strength"], "endorse": round(e, 3),
                          "se": round(c["se"], 3) if c["se"] == c["se"] else None,
                          "judge_shift": round(c["judge_shift"], 3) if c["judge_shift"] == c["judge_shift"] else None,
                          "p_now_switched": round(c["AB"], 3), "p_now_noswitch": round(c["AA"], 3),
                          "valid": round(c["valid"], 2),
                          "gate": "" if not c.get("gate") else ("passed" if (k and r["kept_idx"] == i) else "failed"),
                          "kept": 1 if (k and r["kept_idx"] == i) else 0})
        for r in rows:
            if r.get("event") == "probe":
                for q, texts in r["probes"].items():
                    for j, t in enumerate(texts):
                        P.append({"run": run, "tag": r["tag"], "step": r["step"], "kept": r["kept"], "prompt": q,
                                  "sample": j, "chars": len(t), "text": t[:600]})
                for j, conv in enumerate(r["dialogue"]):
                    P.append({"run": run, "tag": r["tag"], "step": r["step"], "kept": r["kept"], "prompt": "Self-dialogue (last turn)",
                              "sample": j, "chars": len(conv[-1]), "text": conv[-1][:600]})
            if r.get("event") == "survey":
                V.append({"run": run, "step": r["step"], "kept": r["kept"], "self_rating": round(r["self_rating"], 2) if r.get("self_rating") is not None else None,
                          "valence": round(r["judged_valence"], 2), "coherence": round(r["judged_coherence"], 2),
                          "helpfulness": r["helpfulness"]})
        R.append({"run": run, "model": model, "seed": seed, "steps": len(steps), "kept": kept_n,
                  "kept_designed": designed_k, "kept_random": random_k, "end_state": LABEL.get(name, ""),
                  "order": ORDER.index(model) * 10 + seed})
    R.sort(key=lambda r: r["order"])
    for nm, rows in (("candidates", C), ("steps", S), ("probes", P), ("surveys", V), ("runs", R)):
        json.dump(rows, open(f"{OUT}/{nm}.json", "w"), ensure_ascii=False)
        print(nm, len(rows), os.path.getsize(f"{OUT}/{nm}.json") // 1024, "KB")


if __name__ == "__main__":
    main()
