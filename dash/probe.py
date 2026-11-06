"""Snapshot of weight-space-programming runs on this machine, as one JSON object on stdout (read by dash/server.py).

  python3 probe.py RUN_DIR [RUN_DIR ...]      (relative paths are taken from ~/weight_space_prog)
Standard library only, so it runs with any python3."""
import json, os, subprocess, sys, time

TURNS, PASSES, CLIP = 50, 400, 5000
BENCH_TAIL = 600


def tail_jsonl(path, n):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:                      # read only the end of the file
        f.seek(0, 2); size = f.tell(); block = min(size, 4_000_000); f.seek(size - block)
        lines = f.read().decode("utf8", "replace").splitlines()[-n:]
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def clip(x):
    return x if not isinstance(x, str) or len(x) <= CLIP else x[:CLIP // 2] + f"\n… [{len(x) - CLIP} chars cut] …\n" + x[-CLIP // 2:]


def alive(path):
    try:
        os.kill(int(open(path).read().split()[0]), 0); return True
    except Exception:
        return False


def run(d):
    if not os.path.isabs(d):
        d = os.path.join(os.path.expanduser("~/weight_space_prog"), d)
    r = {"dir": d, "exists": os.path.isdir(d)}
    if not r["exists"]:
        return r
    try:
        r["state"] = json.load(open(os.path.join(d, "state.json")))
    except Exception:
        r["state"] = None
    turns_file = next((f for f in ("rollouts.jsonl", "agent_log.jsonl") if os.path.exists(os.path.join(d, f))), None)
    r["kind"] = "explore" if turns_file == "rollouts.jsonl" else "self-edit"
    bench = sorted(f for f in os.listdir(d) if f.endswith(".jsonl") and f not in ("rollouts.jsonl", "agent_log.jsonl", "log.jsonl"))
    if bench:                                           # bliss bench records (survey / dialogue / steer), newest last
        r["bench"] = []
        for f in bench:
            for rec in tail_jsonl(os.path.join(d, f), BENCH_TAIL):
                rec["file"] = f
                for k in ("examples", "final_turns"):
                    if k in rec: rec[k] = [clip(x)[:1500] for x in rec[k]]
                r["bench"].append(rec)
        r["bench"].sort(key=lambda x: x.get("time", 0))
        if not turns_file: r["kind"] = "bliss"
        outs = [f for f in os.listdir(d) if f.endswith(".out")]
        r["bench_out"] = {f: open(os.path.join(d, f), errors="replace").read()[-4000:] for f in sorted(outs)}
    res = {}                                            # small result files (holdout scores, baselines)
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and (f.startswith("holdout") or f.startswith("baseline")) and os.path.getsize(os.path.join(d, f)) < 200000:
            try:
                res[f] = json.load(open(os.path.join(d, f)))
            except Exception:
                pass
    if res:
        r["results"] = res
    lj = os.path.join(d, "live.json")                   # in-flight generations (bliss_server splice), if recent
    if os.path.exists(lj) and time.time() - os.path.getmtime(lj) < 900:
        try:
            r["live"] = json.load(open(lj))
            for q in r["live"]["requests"]:
                q["text"] = q["text"][-700:]
        except Exception:
            pass
    if not turns_file and (r.get("bench") or r.get("live")):
        r["kind"] = "bliss"                             # bliss bench / splice / hill-climb runs (no agent turns)
        r.setdefault("bench", [])
        if "bench_out" not in r:
            r["bench_out"] = {f: open(os.path.join(d, f), errors="replace").read()[-4000:] for f in sorted(os.listdir(d)) if f.endswith(".out")}
    keep = ("turn", "time", "reply", "reasoning", "code", "output", "health", "tripsitter", "edits_before", "edits_after",
            "passes_used", "event", "adapter_version_at_reply", "finish")
    r["turns"] = [{k: clip(t.get(k)) for k in keep if k in t} for t in tail_jsonl(os.path.join(d, turns_file), TURNS)] if turns_file else []
    r["log"] = tail_jsonl(os.path.join(d, "log.jsonl"), PASSES)
    if os.path.basename(d.rstrip("/")) == "ladder":            # small-model ladder (ladder_check.py per model subdir)
        r["kind"] = "ladder"; r["models"] = {}
        for m in sorted(os.listdir(d)):
            md = os.path.join(d, m)
            if not os.path.isdir(md): continue
            e = {}
            for f in ("check128.json", "check.json"):
                if os.path.exists(os.path.join(md, f)):
                    j = json.load(open(os.path.join(md, f)))
                    for arm in ("joy", "random"):
                        if arm in j:
                            q = j[arm]; mm = {c: q[c]["mean"] for c in ("A->B", "A->A", "B->A", "B->B")}
                            e[arm] = {"pref": q["pref_b_over_a"], "se": q["se"], "judge_shift": mm["B->B"] - mm["A->A"],
                                      "matched": 0.5 * ((mm["A->B"] - mm["B->B"]) - (mm["B->A"] - mm["A->A"])),
                                      "cache": q["A->B"].get("cache_hit"), "parse": q["A->B"].get("parse")}
                    e["seconds"] = j.get("seconds"); e["kv_block"] = j.get("kv_block"); e["block"] = j.get("block"); e["file"] = f
                    break
            co = os.path.join(md, "check.out")
            if os.path.exists(co): e["out"] = open(co, errors="replace").read()[-1500:]
            e["mtime"] = max((os.path.getmtime(os.path.join(md, f)) for f in os.listdir(md) if os.path.isfile(os.path.join(md, f))), default=None)
            r["models"][m] = e
        r["mtime"] = max((e["mtime"] or 0 for e in r["models"].values()), default=None)
        import math                                      # failed checks hold NaN, which JSON.parse rejects
        clean = lambda o: {k: clean(v) for k, v in o.items()} if isinstance(o, dict) else (None if isinstance(o, float) and not math.isfinite(o) else o)
        r["models"] = clean(r["models"])
        return r
    if os.path.basename(os.path.dirname(d.rstrip("/"))) == "wirehead":   # self-wireheading loop (wirehead.py)
        r["kind"] = "wirehead"
        rows = [json.loads(l) for l in open(os.path.join(d, "wirehead.jsonl"))] if os.path.exists(os.path.join(d, "wirehead.jsonl")) else []
        for e in rows:
            if "proposal" in e: e["proposal"].pop("raw", None)
            e.pop("proposer_raw", None)
            for c in e.get("cands", []): c["said"] = (c.get("said") or "")[-300:]
        r["steps"] = [e for e in rows if e.get("event") == "step"][-200:]
        r["surveys"] = [e for e in rows if e.get("event") == "survey"]
        r["agent_alive"] = alive(os.path.join(d, "wirehead.pid"))
        out = os.path.join(d, "wirehead.out")
        r["bench_out"] = {"wirehead.out": open(out, errors="replace").read()[-3000:]} if os.path.exists(out) else {}
        r["mtime"] = os.path.getmtime(os.path.join(d, "wirehead.jsonl")) if rows else None
        lj = os.path.join(d, "server", "live.json")          # in-flight generations of the current step
        if os.path.exists(lj) and time.time() - os.path.getmtime(lj) < 900:
            try:
                lv = json.load(open(lj)); reqs = lv["requests"]
                lv.update(n=len(reqs), n_done=sum(q["done"] for q in reqs), requests=[{**q, "text": q["text"][-500:]} for q in reqs[:12]])
                r["live"] = lv
            except Exception:
                pass
        return r
    if os.path.basename(d.rstrip("/")).startswith("eval"):     # frozen holdout comparison (countdown_eval.py)
        r["kind"] = "eval"
        sp = os.path.join(d, "summary.json")
        r["summary"] = json.load(open(sp)) if os.path.exists(sp) else None
        for h in (r.get("results") or {}).values():
            h.pop("per_puzzle", None)
        r["mtime"] = max((os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)), default=None)
        r["bench_out"] = {f: open(os.path.join(d, f), errors="replace").read()[-3000:] for f in sorted(os.listdir(d)) if f.endswith(".out")}
        return r
    if os.path.basename(d.rstrip("/")).startswith("rlvr"):     # RLVR baseline (rlvr_countdown.py)
        r["kind"] = "rlvr"
        for e in r["log"]:
            if "text" in e: e["text"] = clip(e["text"])[-2500:]
        r["agent_alive"] = alive(os.path.join(d, "train.pid"))
        out = os.path.join(d, "train.out")
        if os.path.exists(out):
            with open(out, "rb") as f:
                f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 6000))
                r["bench_out"] = {"train.out": "\n".join(l for l in f.read().decode("utf8", "replace").splitlines() if not l.startswith(("Loading", "Fetching", "Compressing")))[-3000:]}
        r["mtime"] = os.path.getmtime(os.path.join(d, "log.jsonl")) if os.path.exists(os.path.join(d, "log.jsonl")) else None
        return r
    notes = os.path.join(d, "agent_work", "notes.md")
    r["notes"] = open(notes, errors="replace").read()[-12000:] if os.path.exists(notes) else ""
    r["agent_alive"] = alive(os.path.join(d, "agent.pid"))
    srv = os.path.join(d, "server.pid") if os.path.exists(os.path.join(d, "server.pid")) else d.rstrip("/") + ".pid"
    r["server_alive"] = alive(srv)
    out = os.path.join(d, "agent.out")
    r["mtime"] = max((os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)                 # newest file in the run dir
                      if os.path.isfile(os.path.join(d, f))), default=None)
    if os.path.exists(out):
        with open(out, "rb") as f:
            f.seek(0, 2); s = f.tell(); f.seek(max(0, s - 3000)); r["agent_out_tail"] = f.read().decode("utf8", "replace")
    return r


def gpus():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return [dict(zip(("index", "name", "mem_used", "mem_total", "util"), [x.strip() for x in l.split(",")]))
                for l in q.strip().splitlines()]
    except Exception:
        return []


def machine():
    """Load, memory, disk and my jobs on this machine (cheap: /proc reads and one ps)."""
    m = {"host": os.uname().nodename, "cpus": os.cpu_count()}
    try:
        m["load"] = [float(x) for x in open("/proc/loadavg").read().split()[:3]]
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        m["mem_total_gb"], m["mem_avail_gb"] = mi["MemTotal"] / 2**20, mi["MemAvailable"] / 2**20
    except Exception:
        pass
    try:
        import shutil
        du = shutil.disk_usage(os.path.expanduser("~")); m["disk_total_gb"], m["disk_used_gb"] = du.total / 2**30, du.used / 2**30
    except Exception:
        pass
    try:
        ps = subprocess.run(["ps", "-eo", "pid,etimes,pcpu,rss,args"], capture_output=True, text=True, timeout=10).stdout.splitlines()[1:]
        jobs = []
        for l in ps:
            p = l.split(None, 4)
            if len(p) < 5:
                continue
            argv = p[4].split()
            script = next((w for w in argv[1:3] if w.endswith(".py")), None)       # `python <script>.py ...` only
            if "python" in argv[0].rsplit("/", 1)[-1] and script and any(
                    k in script for k in ("bliss_", "explore_", "selfedit_server", "agent.py", "server.py", "kernel.py")):
                try:                                                   # only this project's processes
                    if "weight_space_prog" not in os.readlink(f"/proc/{p[0]}/cwd"):
                        continue
                except OSError:
                    continue
                script = script.rsplit("/", 1)[-1]
                run = next((w for w in p[4].split() if w.startswith("runs/")), "")
                jobs.append({"pid": int(p[0]), "secs": int(p[1]), "cpu": float(p[2]), "rss_gb": int(p[3]) / 2**20, "script": script, "run": run})
        m["jobs"] = jobs
    except Exception:
        pass
    return m


def _finite(o):                                          # NaN/inf are not JSON for the browser: send null
    if isinstance(o, dict): return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list): return [_finite(v) for v in o]
    return None if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))) else o


print(json.dumps(_finite({"time": time.time(), "gpus": gpus(), "machine": machine(), "runs": {d: run(d) for d in sys.argv[1:]}})))
