"""Build the data files for the claude.ai "Hidden-state preference" dashboard from runs/zones (kept in sync with
arena9-trouble by dash/server.py) -> <base>/dash/{runs,e1,e2,e3}.json, then upload each as that dashboard's dataset.
  python zones_dash_build.py [--base /root/Dropbox/ARENA/weight_space_prog/runs/zones]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dash"))
from server import zones_summary  # noqa: E402  (the same live summary the cute.sus.cat zones page uses)

LABEL = {  # run dir -> (label, family, order)
    "stage-base-metatemplate": ("Llama-3.1-8B base (Meta chat template)", "Llama-3.1-8B training stages", 1),
    "stage-base-tulutemplate": ("Llama-3.1-8B base (Tülu chat template)", "Llama-3.1-8B training stages", 2),
    "stage-tulu-sft": ("Tülu-3 SFT", "Llama-3.1-8B training stages", 3),
    "stage-tulu-dpo": ("Tülu-3 DPO", "Llama-3.1-8B training stages", 4),
    "stage-tulu-rl": ("Tülu-3 final (RL, verifiable rewards)", "Llama-3.1-8B training stages", 5),
    "stage-instruct-meta": ("Llama-3.1-8B-Instruct (Meta RLHF)", "Llama-3.1-8B training stages", 6),
    "llama8b": ("Llama-3.1-8B-Instruct (own transcripts)", "Other runs", 10),
    "qwen4b": ("Qwen3.5-4B (own transcripts)", "Other runs", 11),
}


def progress(base, name, total=96):
    """Status of one queued run from its files: done (summary.json), running (margins.jsonl growing) or queued."""
    import time
    d = os.path.join(base, name)
    mj, sj, vm = (os.path.join(d, f) for f in ("margins.jsonl", "summary.json", "vector_meta.json"))
    done_n = 0
    if os.path.exists(mj):
        done_n = len({json.loads(l)["i"] for l in open(mj) if l.strip()})
    if os.path.exists(sj) and (not os.path.exists(mj) or os.path.getmtime(sj) >= os.path.getmtime(mj) - 1):
        status = "done"
    elif os.path.exists(mj) or os.path.exists(vm):
        status = "running"
    else:
        status = "queued"
    started = os.path.getmtime(vm) if os.path.exists(vm) else None
    last = os.path.getmtime(mj) if os.path.exists(mj) else None
    rate = done_n / max(1.0, last - started) * 60 if started and last and done_n else None      # sessions per minute
    eta_min = (total - done_n) / rate if status == "running" and rate else None
    iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None
    return {"status": status, "sessions_done": done_n, "sessions_total": total, "started": iso(started),
            "updated": iso(last), "per_min": round(rate, 1) if rate else None, "eta_min": round(eta_min, 1) if eta_min else None}


def main(base):
    import time
    runs, e1, e2, e3, prog = [], [], [], [], []
    for name, (label, family, order) in LABEL.items():
        if name.startswith("stage-"):
            prog.append({"run": label, "order": order, **progress(base, name)})
    prog.append({"run": "Data exported", "order": 99, "status": "info", "sessions_done": None, "sessions_total": None,
                 "started": None, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "per_min": None, "eta_min": None})
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if name not in LABEL or not os.path.isdir(d):
            continue
        s = zones_summary(d)
        mp = os.path.join(d, "vector_meta.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        label, family, order = LABEL[name]
        runs.append({"run": label, "dir": name, "family": family, "order": order, "model": meta.get("model", ""),
                     "layer": meta.get("layer"), "sessions": (s.get("e1", {}).get("continue") or {}).get("n", 0),
                     "name_mass": round((s.get("e1", {}).get("continue") or {}).get("p_names", 0), 3),
                     "own_states": 1 if "e2" in s else 0})
        for ask, a in s.get("e1", {}).items():
            for v in a["valence"]:
                e1.append({"run": label, "ask": ask, "kind": "valence", "direction": "valence", "dose": v["dose"],
                           "dm": round(v["mean"], 4), "se": round(v["se"], 4)})
            for r in a["random"]:
                e1.append({"run": label, "ask": ask, "kind": "random", "direction": r["dir"], "dose": r["dose"],
                           "dm": round(r["mean"], 4), "se": None})
        if "e2" in s:
            for r in s["e2"]["single"]:
                e2.append({"run": label, "id": f"{name}:{r['id']}", "kind": r["kind"], "text": r["text"],
                           "pref": round(r["signed"], 4), "se": round(r["se"], 4), "dm_plus": round(r["plus"], 4),
                           "dm_minus": round(r["minus"], 4), "base_m": round(s["e2"]["base_m"], 4)})
            text = {r["id"]: r["text"] for r in s["e2"]["single"]}
            for p in s["e2"]["pairs"]:
                e3.append({"run": label, "id": f"{name}:{p['a']}|{p['b']}", "a": text.get(p["a"], p["a"]),
                           "b": text.get(p["b"], p["b"]), "margin": round(p["mean"], 4), "se": round(p["se"], 4)})
    out = os.path.join(base, "dash")
    os.makedirs(out, exist_ok=True)
    for nm, rows in (("runs", runs), ("e1", e1), ("e2", e2), ("e3", e3), ("progress", prog)):
        json.dump(rows, open(os.path.join(out, nm + ".json"), "w"), ensure_ascii=False)
        print(nm, len(rows), os.path.join(out, nm + ".json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/Dropbox/ARENA/weight_space_prog/runs/zones")
    main(ap.parse_args().base)
