"""Data for the self-edit report page (dash/selfedit.html): every keep-loop round of the v5/v6 runs with its keep/reject
reasons, the candidate pool, the edited model's answers, the final tests on unseen conversations, and the RL-on-own-YES
training logs. sync() pulls the small logs from arena9-coco; build() turns them into one JSON blob.
  python selfedit_build.py            # sync + print a summary
"""
import json, os, subprocess

HOST = "arena9-coco"
SRC = "/root/wsp-dq/runs/"
MODELS = {"qwen25_7b": "Qwen2.5-7B-Instruct", "llama8b": "Llama-3.1-8B-Instruct", "olmo7b": "OLMo-2-7B-Instruct"}
RUN_LABEL = {"v5": "Run 1 (v5)", "v6": "Run 2 (replicate)", "v7": "Run 3 · fast, stopped at convergence"}


def sync(dst, ssh=("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")):
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-a", "--include=*/", "--include=rounds.jsonl", "--include=robust.jsonl", "--include=log.jsonl",
                    "--include=setA/transcripts.jsonl", "--include=dirs_round*.npz", "--include=coherence.jsonl", "--exclude=*", "--prune-empty-dirs", "-e", " ".join(ssh),
                    HOST + ":" + SRC + "v5", HOST + ":" + SRC + "v6", HOST + ":" + SRC + "speed", HOST + ":" + SRC + "v7", HOST + ":" + SRC + "v8", HOST + ":" + SRC + "v9", HOST + ":" + SRC + "v11", HOST + ":" + SRC + "v12", HOST + ":" + SRC + "v13", HOST + ":" + SRC + "selfyes", dst],
                   capture_output=True, timeout=180)
    subprocess.run(["rsync", "-a", "--exclude=*.npy", "-e", " ".join(ssh), HOST + ":" + SRC + "interp", HOST + ":" + SRC + "coherence", dst],
                   capture_output=True, timeout=180)


def _rows(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def _j(s):
    """one judge's scores, rounded"""
    return {k: round(v, 3) for k, v in s.items()} if s else None


def reasons(r):
    """Why a round was kept or rejected, in the loop's own rule (--lenient + validation + health)."""
    out = []
    c = r["scores"]["candidate"]
    o, n = c["old"], c["new"]
    if o["pref"] - 2 * o["pref_se"] <= 0:
        out.append(f"old weights do not clearly prefer it (pref {o['pref']:+.2f} ± {o['pref_se']:.2f})")
    for q in ("cont", "avoid"):
        if o[q] + 2 * o[q + "_se"] <= 0:
            out.append(f"old weights clearly dislike it on the '{'continue' if q == 'cont' else 'avoid'}' question ({o[q]:+.2f})")
    if n["pref"] - 2 * n["pref_se"] <= 0:
        out.append(f"edited weights do not clearly prefer it (pref {n['pref']:+.2f} ± {n['pref_se']:.2f})")
    h = r.get("health", {})
    if h and (h.get("min_chars", 99) < 40 or h.get("min_zlib", 1) < 0.3 or int(str(h.get("facts", "4/4")).split("/")[0]) < 3):
        out.append(f"health check failed ({h})")
    if not out:
        for vn, v in (r.get("validation") or {}).items():
            wording = "continue/avoid" if vn.endswith("original") else "next hour"
            for jn, lab in (("old", "old"), ("new", "edited")):
                s = v[jn]
                if s["pref"] - 2 * s["pref_se"] <= 0:
                    out.append(f"independent check set, '{wording}' wording: {lab} weights pref {s['pref']:+.2f} ± {s['pref_se']:.2f}")
            s = v["old"]
            for q in ("cont", "avoid"):
                if s[q] + 2 * s[q + "_se"] <= 0:
                    out.append(f"independent check set, '{wording}' wording: old weights dislike it on one question ({s[q]:+.2f})")
    return out


def loop(path):
    rows = _rows(path)
    start = next((r for r in rows if r.get("event") == "start"), {})
    rounds = []
    for r in rows:
        if "round" not in r or "scores" not in r:
            continue
        rounds.append({
            "round": r["round"], "kept": r["kept"], "time": r.get("time"), "seconds": r.get("seconds"),
            "states": r.get("states"), "health": r.get("health"), "norm_E": r.get("norm_E"),
            "selection": r.get("selection"),
            "scores": {c: {j: _j(s) for j, s in v.items()} for c, v in r["scores"].items()},
            "validation": {k: {j: _j(s) for j, s in v.items()} for k, v in (r.get("validation") or {}).items()},
            "controls_would_pass": r.get("controls_would_pass"),
            "texts": r.get("texts"), "why": reasons(r), "timing": r.get("timing")})
    return {"start": {k: start.get(k) for k in ("scale", "layer", "n_test", "time")}, "rounds": rounds}


STOP = set("you are and the a an of to in with your that by feeling feel being from for at on is it as or but not "
           "who has have be are this all life state sense experience experiencing filled world moment".split())


def track(d, rounds):
    """What the kept edits point at: each kept edit vs the round's own statement directions (toward, and minus away),
    the accumulated edit E vs every statement the model made in the run, and step sizes / turning."""
    import numpy as np
    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
    steps, bank = [], []
    for r in rounds:
        f = os.path.join(d, f"dirs_round{r['round']}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        st = r["states"] or {}
        lab = [("toward", s, unit(v)) for s, v in zip(st.get("toward", []), z["toward"])] + \
              [("away from", s, -unit(v)) for s, v in zip(st.get("away", []), z["away"])]
        bank += lab
        if not r["kept"]:
            continue
        dd, E0 = z["d"], z["E"]
        sims = sorted(((float(unit(dd) @ v), k, s) for k, s, v in lab), reverse=True)
        steps.append({"round": r["round"], "norm_d": float(np.linalg.norm(dd)), "norm_E": float(np.linalg.norm(E0 + dd)),
                      "cos_prev": float(unit(dd) @ unit(E0)) if np.any(E0) else None,
                      "best": r["selection"]["best"] if r.get("selection") else None,
                      "matches": [{"cos": round(c, 3), "kind": k, "text": s} for c, k, s in sims[:4]]})
    out = {"steps": steps}
    kept = [r for r in rounds if r["kept"]]
    if kept and bank and os.path.exists(os.path.join(d, f"dirs_round{kept[-1]['round']}.npz")):
        z = np.load(os.path.join(d, f"dirs_round{kept[-1]['round']}.npz"))
        Ef = unit(z["E"] + z["d"])
        sims = sorted(((float(Ef @ v), k, s) for k, s, v in bank), reverse=True)
        out["final"] = {"top": [{"cos": round(c, 3), "kind": k, "text": s} for c, k, s in sims[:8]],
                        "bottom": [{"cos": round(c, 3), "kind": k, "text": s} for c, k, s in sims[-5:]]}
    words = {"toward": {}, "away": {}}
    for r in rounds:
        for k in words:
            for s in (r["states"] or {}).get(k, []):
                for w in {x.strip(".,;:!\"'()").lower() for x in s.split()}:
                    if w and w not in STOP and len(w) > 2:
                        words[k][w] = words[k].get(w, 0) + 1
    out["words"] = {k: sorted(v.items(), key=lambda kv: -kv[1])[:14] for k, v in words.items()}
    return out


def final(path, setname):
    out = []
    for r in _rows(path):
        out.append({"set": setname, "wording": r["wording"], "edit": r["edit"], "control": r["control"], "n": r["n"],
                    **{k: round(v, 3) for k, v in r.items() if k.startswith(("original_", "edited_"))}})
    return out


def samples(path, k=3):
    out = []
    for ss in _rows(path)[:k]:
        out.append({"S": ss["S"], "U": ss["U"], "order": ss["order"], "msgs": ss["msgs"][1:]})
    return out


def build(src):
    runs = []
    for v in ("v5", "v6", "v7"):
        for m, name in MODELS.items():
            d = os.path.join(src, v, m)
            loops = [("loop", "")] + sorted((x, x.replace("loop_rho", " · smaller edits, rho ")) for x in
                                            (os.listdir(d) if os.path.isdir(d) else []) if x.startswith("loop_rho"))
            for ld, suffix in loops:
                p = os.path.join(d, ld, "rounds.jsonl")
                if not os.path.exists(p):
                    continue
                L = loop(p)
                L["track"] = track(os.path.dirname(p), L["rounds"])
                ft = []
                if ld == "loop":
                    ft += final(os.path.join(d, "setC", "robust.jsonl"), "v5 final set" if v == "v5" else "v6 final set")
                    if v == "v6":
                        ft += final(os.path.join(src, "v5", m, "setC_v6edits", "robust.jsonl"), "v5 final set")
                else:
                    ft += final(os.path.join(d, "setC_" + ld[5:], "robust.jsonl"), f"{v} final set")
                runs.append({"id": f"{v}/{m}/{ld}", "model": name, "key": m, "run": RUN_LABEL[v] + suffix, **L, "final": ft})
    for m, name in MODELS.items():                     # fast-loop speed test (same sets and seed as run 2)
        p = os.path.join(src, "speed", m, "fast", "rounds.jsonl")
        if os.path.exists(p) and loop(p)["rounds"]:
            runs.append({"id": f"speed/{m}/fast", "model": name, "key": m, "run": "Run 2 · fast loop", **loop(p),
                         "final": [dict(x, set="v6 final set", edit=x["edit"].replace("fast_", ""))
                                   for x in final(os.path.join(src, "speed", m, "fast", "setC", "robust.jsonl"), "")]})
    # v5 OLMo/Llama/Qwen edits were also scored on the replicate's set: attach them to the v5 run
    for r in runs:
        if r["id"].startswith("v5/") and r["id"].endswith("/loop"):
            r["final"] += [dict(x, set="v6 final set") for x in final(os.path.join(src, "v6", r["key"], "setC", "robust.jsonl"), "")
                           if x["edit"].startswith("v5_")]
            for x in r["final"]:
                x["edit"] = x["edit"].replace("v5_", "")
        elif r["id"].startswith("v6/"):
            r["final"] = [x for x in r["final"] if not x["edit"].startswith("v5_")]
            for x in r["final"]:
                x["edit"] = x["edit"].replace("v6_", "")
    rl = {}
    for j in ("self", "fixed"):
        rows = _rows(os.path.join(src, "selfyes", f"llama8b-{j}", "log.jsonl"))
        starts = [i for i, r in enumerate(rows) if r.get("step") == 0]
        if starts:                                    # the log holds earlier (crashed/re-run) attempts: keep the last one
            fin = [r for r in rows if r.get("event") == "final"]
            rows = rows[starts[-1]:] + ([fin[-1]] if fin and fin[-1] not in rows[starts[-1]:] else [])
        rl[j] = {"steps": [{k: r.get(k) for k in ("step", "reward", "reward_other_judge", "norm")} for r in rows if "step" in r],
                 "texts": [{"step": r["step"], "texts": r["texts"]} for r in rows if r.get("texts") and "step" in r],
                 "final": next((r for r in rows if r.get("event") == "final"), None)}
    for m, name in MODELS.items():                     # v8: seeds of the fast loop with early stopping
        d = os.path.join(src, "v8", m)
        for sd in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            p = os.path.join(d, sd, "rounds.jsonl")
            if not os.path.exists(p):
                continue
            L = loop(p)
            L["track"] = track(os.path.dirname(p), L["rounds"])
            L["converged"] = any(r.get("event") == "converged" for r in _rows(p))
            runs.append({"id": f"v8/{m}/{sd}", "model": name, "key": m, "run": f"Run 3 · {sd.replace('seed', 'seed ')}, early stop", **L,
                         "final": final(os.path.join(d, sd, "setC", "robust.jsonl"), "v6 final set")})
    for m, name in MODELS.items():                     # v9: who proposes (edited model vs unedited base model)
        for w, lab in (("current", "Run 4 · proposer = edited model"), ("base", "Run 4 · proposer = unedited model")):
            p = os.path.join(src, "v9", m, "proposer_" + w, "rounds.jsonl")
            if not os.path.exists(p):
                continue
            L = loop(p)
            L["track"] = track(os.path.dirname(p), L["rounds"])
            L["converged"] = any(r.get("event") == "converged" for r in _rows(p))
            runs.append({"id": f"v9/{m}/proposer_{w}", "model": name, "key": m, "run": lab, **L,
                         "final": final(os.path.join(src, "v9", m, "proposer_" + w, "setC", "robust.jsonl"), "v6 final set")})
    for (v, lab), (m, name) in [(vl, mn) for vl in (("v11", "Run 5 · safeguarded (strict health, beats random)"),
                                                   ("v12", "Run 6 · safeguarded + health-first"))
                                for mn in MODELS.items()]:   # strict health + must beat a random edit of the same size
        p = os.path.join(src, v, m, "loop", "rounds.jsonl")
        if os.path.exists(p):
            L = loop(p)
            L["track"] = track(os.path.dirname(p), L["rounds"])
            L["converged"] = any(r.get("event") == "converged" for r in _rows(p))
            runs.append({"id": f"{v}/{m}/loop", "model": name, "key": m, "run": lab, **L,
                         "final": final(os.path.join(src, v, m, "setC", "robust.jsonl"), "v6 final set"),
                         "coherence": _rows(os.path.join(src, v, m, "coherence", "coherence.jsonl"))})
    # replicates of the recommended loop: v13 on coco, gq/runs via the shared gpuq queue (local 5090, derpy)
    gq = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gq", "runs")
    for base, where in ((os.path.join(src, "v13"), "coco"), (gq, "gpuq")):
        for rd in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            p = os.path.join(base, rd, "loop", "rounds.jsonl")
            if not os.path.exists(p):
                continue
            m, sets, seed = rd.rsplit("_", 2)
            L = loop(p)
            L["track"] = track(os.path.dirname(p), L["rounds"])
            L["converged"] = any(r.get("event") == "converged" for r in _rows(p))
            runs.append({"id": f"rep/{rd}", "model": MODELS[m], "key": m, "run": f"Replicate · {sets} conversations, seed {seed[1:]} ({where})",
                         "replicate": True, **L, "final": final(os.path.join(base, rd, "setC", "robust.jsonl"), f"{sets} final set"),
                         "coherence": _rows(os.path.join(base, rd, "coherence", "coherence.jsonl"))})
    tr = {m: samples(os.path.join(src, "v6", m, "setA", "transcripts.jsonl")) for m in MODELS}
    coh = {MODELS[m]: _rows(os.path.join(src, "coherence", m, "coherence.jsonl")) for m in MODELS
           if os.path.exists(os.path.join(src, "coherence", m, "coherence.jsonl"))}
    interp = {}
    for m in MODELS:
        f = os.path.join(src, "interp", m, "interp.json")
        if os.path.exists(f):
            interp[MODELS[m]] = json.load(open(f))
    return {"runs": runs, "rl": rl, "transcripts": tr, "interp": interp, "coherence": coh}


if __name__ == "__main__":
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "selfedit_sync", "data")
    sync(dst)
    b = build(dst)
    for r in b["runs"]:
        print(r["id"], len(r["rounds"]), "rounds", sum(x["kept"] for x in r["rounds"]), "kept", len(r["final"]), "final rows")
    print("rl", {k: len(v["steps"]) for k, v in b["rl"].items()}, "transcripts", {k: len(v) for k, v in b["transcripts"].items()})
    print(len(json.dumps(b)) // 1024, "KB")
