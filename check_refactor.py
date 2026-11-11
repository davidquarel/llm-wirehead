"""Check that the faster keep_scores (cache crop, several wordings from one pass) gives the same numbers as the old one
(deepcopy of the cache, one wording per pass).  python check_refactor.py  (needs guided2_old.py next to it)"""
import argparse
import numpy as np
import guided2, guided2_old
from guided import H
from guided2 import ASK, HOUR

a = argparse.Namespace(model="unsloth/Llama-3.1-8B-Instruct", layer=None)
h = H(a)
path = "runs/v6/llama8b/setB/transcripts.jsonl"
E = np.zeros(h.D, np.float32)
d = np.load("runs/v6/llama8b/loop/E_round1.npy").astype(np.float32)
new = guided2.keep_scores(h, guided2.sessions(h, path, 0, 6, wordings={"o_": ASK, "h_": HOUR}), E, {"c": d}, {"old": E, "new": d})
for w, at in (("o_", ASK), ("h_", HOUR)):
    old = guided2_old.keep_scores(h, guided2_old.sessions(h, path, 0, 6, at), E, {"c": d}, {"old": E, "new": d})
    for j in ("old", "new"):
        print(w, j, "old code", {k: round(v, 4) for k, v in old["c"][j].items()})
        print(w, j, "new code", {k: round(v, 4) for k, v in new[w]["c"][j].items()}, flush=True)
