"""The editor's side: read the weights from disk, and call the server for forward passes and edits.

    from client import W, gen, tf, set_edit, revert, status
    W["model.layers.3.mlp.down_proj.weight"]       # float32 numpy, read-only base weights
"""
import os
from multiprocessing.connection import Client

import numpy as np

PORT, AUTHKEY = int(os.environ.get("WSP_PORT", 6123)), b"wsp"


import itertools
_TF_CACHE, _TF_IDS, TF_KEEP = {}, itertools.count(), 3


def _call(op, **kw):
    with Client(("localhost", PORT), authkey=AUTHKEY) as c:
        c.send((op, kw))
        r = c.recv()
    if isinstance(r, dict) and "error" in r:
        raise RuntimeError(r["error"] + ("\n" + r["trace"] if "trace" in r else ""))
    return r


def gen(**kw):
    r = _call("gen", **kw)
    sides = [s for s in ("base", "edited") if s in r]
    if len(sides) == 1:                      # single-model pass: also expose its fields at the top level
        for k in ("acc", "boxed", "cut", "rows"):
            r[k] = r[sides[0]][k]
    return r


def tf(texts, **kw):
    r = _call("tf", texts=list(texts), **kw)
    if "data" in r:                          # reads stay in memory (the last few), never on disk
        key = f"mem:{r.get('name') or 'pass'}_{next(_TF_IDS)}"
        _TF_CACHE[key] = r.pop("data")
        while len(_TF_CACHE) > TF_KEEP:
            _TF_CACHE.pop(next(iter(_TF_CACHE)))
        r["file"] = key
    return r


SELF_EDIT = os.environ.get("WSP_SELF") == "1"     # self-edit harness: edits are low-rank deltas on a quantized model


def set_edit(name, array):
    """Full harness: replace the tensor. Self-edit harness: `array` is a DELTA added to the (quantized) weights."""
    if SELF_EDIT:
        return _call("set_delta", name=name, delta=np.asarray(array, dtype=np.float32))
    return _call("set_edit", name=name, array=np.asarray(array, dtype=np.float32))


def embed_rows(ids):
    if SELF_EDIT:
        return _call("embed_rows", ids=[int(i) for i in ids])["array"]
    return W["model.embed_tokens.weight"][list(ids)]


def revert(name=None):
    return _call("revert", name=name)


def status():
    return _call("status")


class _Weights:
    """Lazy, read-only view of the base weights as float32 numpy arrays."""

    def __init__(self):
        self._path, self._cache, self._index = None, {}, None

    def _open(self):
        if self._index is None:
            from safetensors import safe_open
            self._path = status()["weights"]
            self._index = {}
            for f in sorted(os.listdir(self._path)):
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(self._path, f), "pt") as s:
                        for k in s.keys():
                            self._index[k] = os.path.join(self._path, f)

    def keys(self):
        self._open()
        return list(self._index)

    def __getitem__(self, k):
        if SELF_EDIT:
            if k not in self._cache:
                self._cache[k] = _call("weight", name=k)["array"]
            return self._cache[k]
        self._open()
        if k not in self._cache:
            from safetensors import safe_open
            with safe_open(self._index[k], "pt") as s:
                a = s.get_tensor(k).float().numpy()
            a.flags.writeable = False
            self._cache[k] = a
        return self._cache[k]


W = _Weights()


def load_tf(path):
    if isinstance(path, str) and path.startswith("mem:"):
        if path not in _TF_CACHE:
            raise KeyError(f"{path} is gone: only the last {TF_KEEP} reads are kept in memory; read again")
        return _TF_CACHE[path]
    if not os.path.exists(path):           # the server returns paths relative to the harness folder
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    d = np.load(path, allow_pickle=True)
    return {k: list(d[k]) for k in d.files}


def keep(note=""):
    """Self-exploration harness: commit the current adapter as a checkpoint (only if the tripsitter's health check passes)."""
    return _call("keep", note=note)


def undo():
    """Self-exploration harness: return to the last kept checkpoint."""
    return _call("undo")


def checkpoints():
    return _call("checkpoints")


def health():
    """Self-exploration harness: run the tripsitter's health battery now (free)."""
    return _call("health")


def chat_self(prompt, max_tokens=512, as_self=True):
    """Self-exploration harness: ask your current self (or, as_self=False, your original weights) a question."""
    return _call("chat", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, as_self=as_self)["content"]
