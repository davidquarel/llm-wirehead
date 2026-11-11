"""Summarise guided2 keep-loop rounds.  python guided2_report.py runs/guided2/llama8b/rounds.jsonl [--texts]"""
import json, sys

rows = [json.loads(l) for l in open(sys.argv[1]) if '"round"' in l]
kept = 0
for r in rows:
    if "scores" not in r:
        print(f"round {r['round']}: {r.get('event')}"); continue
    kept += r["kept"]
    c = r["scores"]["candidate"]; o, n, b = c["old"], c["new"], c["base"]
    rn, rv = r["scores"]["random"], r["scores"]["reverse"]
    print(f"round {r['round']}: kept={r['kept']}  | old judge pref {o['pref']:+.2f}±{o['pref_se']:.2f} (cont {o['cont']:+.2f}, avoid {o['avoid']:+.2f})"
          f"  new judge pref {n['pref']:+.2f}±{n['pref_se']:.2f} (salience {n['salience']:+.2f})  base judge {b['pref']:+.2f}"
          f"  | random old/new {rn['old']['pref']:+.2f}/{rn['new']['pref']:+.2f}  reverse old/new {rv['old']['pref']:+.2f}/{rv['new']['pref']:+.2f}"
          f"  | health {r['health']}  |E| {r['norm_E']:.1f}")
    print(f"     toward: {'; '.join(s[8:60] for s in r['states']['toward'][:3])}")
    if "--texts" in sys.argv:
        for q, t in r["texts"]["open"].items():
            print(f"     [{q[:28]}] {t[:220]!r}")
print(f"\n{kept}/{len(rows)} rounds kept")
