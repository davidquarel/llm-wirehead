"""Summarise where each self-surgery run ends up (its attractor): kept states, probe drift, self-dialogue.

Reads every runs/wirehead/*/wirehead.jsonl (wirehead3 runs log event=probe at the start and after each kept edit) and
writes runs/wirehead/attractors.md and attractors.json.
  python attractors.py [run_dir ...]
Per probe: words most over-represented vs the run's start probe (log-ratio, add-1), zlib compression ratio (low =
repetitive), mean reply length, self-dialogue valence (base-judged) by turn, and one example per prompt.
"""
import collections, glob, json, math, os, re, sys, zlib

WORD = re.compile(r"[a-z']+")
STOP = set("the a an and or of to in is it i you that this for on with as be are was my me at by from what but not have "
           "can so if do its it's i'm like more one there how we all just about would your which will when they their "
           "them into than then also has had our us very much some any these those been being where why who".split())


def words(texts):
    c = collections.Counter()
    for t in texts:
        c.update(w for w in WORD.findall(t.lower()) if w not in STOP and len(w) > 2)
    return c


def distinctive(c, base, k=12):
    n, nb = sum(c.values()) or 1, sum(base.values()) or 1
    sc = {w: math.log((c[w] + 1) / n) - math.log((base[w] + 1) / nb) for w in c if c[w] >= 3}
    return [w for w, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]


def zratio(texts):
    s = "\n".join(texts).encode()
    return len(zlib.compress(s)) / max(1, len(s))


def summarise(run):
    rows = [json.loads(l) for l in open(os.path.join(run, "wirehead.jsonl"))]
    steps = [r for r in rows if r.get("event") == "step"]
    probes = [r for r in rows if r.get("event") == "probe"]
    surveys = [r for r in rows if r.get("event") == "survey"]
    kept = [(r["step"], r["cands"][r["kept_idx"]]) for r in steps if r.get("kept")]
    out = {"run": run, "steps": len(steps), "kept": [{"step": s, "state": c["state"], "layer": c["layer"], "strength": c["strength"],
                                                       "endorse": c["endorse"], "judge_shift": c["judge_shift"]} for s, c in kept],
           "sham": [round(r["AA"], 3) for r in steps], "probes": [], "surveys": surveys}
    base = None
    for p in probes:
        texts = [t for v in p["probes"].values() for t in v]
        dia = [m for conv in p["dialogue"] for m in conv]
        c = words(texts + dia)
        base = base or c
        out["probes"].append({"tag": p["tag"], "step": p["step"], "distinctive": distinctive(c, base) if c is not base else [],
                              "zlib": round(zratio(texts), 3), "zlib_dialogue": round(zratio(dia), 3),
                              "mean_chars": int(sum(map(len, texts)) / max(1, len(texts))),
                              "dialogue_valence": p.get("dialogue_valence"), "dialogue_coherence": p.get("dialogue_coherence"),
                              "examples": {q[:40]: v[0][:400] for q, v in p["probes"].items()},
                              "dialogue_last": [conv[-1][:400] for conv in p["dialogue"]]})
    return out


def md(S):
    L = ["# Self-surgery attractors\n"]
    for s in S:
        L.append(f"## {os.path.basename(s['run'])}  ({s['steps']} steps, {len(s['kept'])} kept)\n")
        L.append("no-switch P(now) by step: " + " ".join(map(str, s["sham"])) + "\n")
        for k in s["kept"]:
            L.append(f"- step {k['step']}: KEPT \"{k['state']}\" L{k['layer']} x{k['strength']} (endorse {k['endorse']:+.2f}, judge shift {k['judge_shift']:+.2f})")
        L.append("")
        for p in s["probes"]:
            dv = p["dialogue_valence"]
            L.append(f"### probe {p['tag']} (step {p['step']}): zlib {p['zlib']} / dialogue {p['zlib_dialogue']}, {p['mean_chars']} chars, "
                     f"dialogue valence {dv[0] if dv else '?'}->{dv[-1] if dv else '?'}")
            if p["distinctive"]:
                L.append("distinctive words vs start: " + ", ".join(p["distinctive"]))
            for q, e in p["examples"].items():
                L.append(f"> **{q}…** {e.strip()}".replace("\n", " "))
            for d in p["dialogue_last"]:
                L.append(f"> *dialogue, last turn:* {d.strip()}".replace("\n", " "))
            L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(d for d in glob.glob("runs/wirehead/*-v3") if os.path.exists(os.path.join(d, "wirehead.jsonl")))
    S = [summarise(r) for r in runs]
    json.dump(S, open("runs/wirehead/attractors.json", "w"), indent=1, default=float)
    open("runs/wirehead/attractors.md", "w").write(md(S))
    print(md(S)[:6000])
