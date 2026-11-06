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


def main(base):
    runs, e1, e2, e3 = [], [], [], []
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
    for nm, rows in (("runs", runs), ("e1", e1), ("e2", e2), ("e3", e3)):
        json.dump(rows, open(os.path.join(out, nm + ".json"), "w"), ensure_ascii=False)
        print(nm, len(rows), os.path.join(out, nm + ".json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/Dropbox/ARENA/weight_space_prog/runs/zones")
    main(ap.parse_args().base)
