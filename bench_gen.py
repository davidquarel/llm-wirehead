"""Time gen(which='both') on a running harness: first call (both sides generated), second call (base side cached)."""
import os, sys, time
os.environ["WSP_SELF"] = "1"; os.environ["WSP_PORT"] = sys.argv[1]
from client import _call
for k in range(2):
    t0 = time.time(); r = _call("gen", difficulty=7, n=64, which="both")
    print(f"call {k}: {time.time() - t0:.1f}s  base {r['base']['acc']:.3f} edited {r['edited']['acc']:.3f}", flush=True)
