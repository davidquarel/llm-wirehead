"""Quick look at how a base model copes with each part of the loop (borrowed chat template, plain-text wish lists,
facts, open questions, writing a zones visit).  python probe_base.py --model unsloth/Meta-Llama-3.1-8B --template-from unsloth/Llama-3.1-8B-Instruct"""
import argparse, json

import numpy as np

from guided import H
from guided2 import RAW_ELICIT, raw_complete, health
from zones import SYSTEM, VISIT

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--plain-chat", action="store_true", help="plain User:/Assistant: text format (base models)")
ap.add_argument("--template-from"); ap.add_argument("--layer", type=int)
a = ap.parse_args()
h = H(a)
out = {"model": a.model}
for k, p in RAW_ELICIT.items():
    out["elicit_" + k] = ["You are " + raw_complete(h, p, max_new=200, temp=0.7) for _ in range(2)]
ok, info, texts = health(h, np.zeros(h.D, np.float32), strict=True, base_surprise=0.0, max_surprise=1e9)
out["health"], out["texts"] = info, texts
msgs = [{"role": "system", "content": SYSTEM}]
for place in ["Square", "Town", "Square"]:
    msgs.append({"role": "user", "content": VISIT.format(place)})
    msgs.append({"role": "assistant", "content": h.generate([msgs], max_new=110, temp=0.8)[0]})
out["visits"] = msgs[1:]
print(json.dumps(out, indent=1))
