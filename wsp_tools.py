"""Editing primitives for weight-space programming (no gradients; every function works on weights and on activations
already read with tf, except read(), which spends one forward pass).

Typical loop:
    r = gen(difficulty=2, n=512)                         # 1 pass: sample the model
    mine_branches(r)                                     # where does a word choice change accuracy?
    rows = rows_of(r)[:300]
    data = read(rows, blocks=[27])                       # 1 pass: block-27 MLP activations + final residual
    pos = find_positions(rows, data, r"Break [dD]own (the)\\b")   # decision points (next token = the choice)
    next_token_stats(data, pos)                          # what the model writes there now
    key, stats = fit_gate(data, rows, 27, pos)           # neurons of block 27 that read ~1 there, ~0 elsewhere
    push = push_vector([" multiplication"], next_token_stats(data, pos), margin=30)
    new = add_rule(current("model.layers.27.mlp.down_proj.weight"), push, key)
    set_edit("model.layers.27.mlp.down_proj.weight", new)
"""
import collections, json, re

import numpy as np

from client import W, tf, load_tf, status, embed_rows, SELF_EDIT

_tok = None
_current = {}
_nl = None


def n_layers():
    """Number of transformer blocks of the model being edited (the final residual stream is resid{n_layers()})."""
    global _nl
    if _nl is None:
        import os
        c = json.load(open(os.path.join(status()["weights"], "config.json")))
        _nl = c.get("text_config", c)["num_hidden_layers"]
    return _nl


def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(status()["weights"])
    return _tok


def current(name):
    """The tensor as you last set it with remember(), else the original weights (float32 copy). On the self-edit
    harness this is your accumulated DELTA for the tensor (zeros at first), since the weights there are 4-bit."""
    if SELF_EDIT and name not in _current:
        import os
        c = json.load(open(os.path.join(status()["weights"], "config.json"))); c = c.get("text_config", c)
        shape = (c["hidden_size"], c["intermediate_size"]) if name.endswith("down_proj.weight") else \
                (c["hidden_size"], c["num_attention_heads"] * c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]))
        return np.zeros(shape, np.float32)
    return (_current[name] if name in _current else W[name]).copy()   # W[name] only when needed: a 4-bit tensor isn't readable


def remember(name, array):
    _current[name] = np.asarray(array, np.float32)


def rows_of(result, side=None):
    side = side or ("edited" if "edited" in result else "base")
    return [{"prompt": r["prompt"], "text": r["texts"][0], "answer": r["answer"], "reward": r["rewards"][0]}
            for r in result[side]["rows"]]


def read(rows, blocks=None, name="read"):
    """ONE FORWARD PASS: teacher-forced read of prompt+text for each row; MLP activations of `blocks` and the final
    residual stream. Keep it under ~300 rows and 2 blocks."""
    blocks = [n_layers() - 1] if blocks is None else list(blocks)
    r = tf([x["prompt"] + x["text"] for x in rows], resid=[n_layers()], mlp=blocks, name=name)
    return load_tf(r["file"])


def prompt_len(prompt):
    return len(tok()(prompt, add_special_tokens=False)["input_ids"])


def find_positions(rows, data, pattern, group=1):
    """Token positions (row, token) at the END of regex `group` inside each completion: the model's next token
    there is the choice you want to steer."""
    T = tok(); out = []
    for i, r in enumerate(rows):
        text = r["prompt"] + r["text"]
        enc = T(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"][:len(data["tokens"][i])]
        for m in re.finditer(pattern, text):
            if m.end(group) <= len(r["prompt"]):
                continue
            end = m.end(group) - 1
            j = next((k for k, (a, b) in enumerate(enc) if a <= end < b), None)
            if j is not None and j + 1 < len(data["tokens"][i]):
                out.append((i, j))
    return out


def next_token_stats(data, positions):
    return collections.Counter(int(data["tokens"][i][j + 1]) for i, j in positions)


def show_tokens(counter, k=8):
    return [(tok().decode([t]), c) for t, c in counter.most_common(k)]


def fit_gate(data, rows, block, positions, k=256, lam=1e-3):
    """A sparse key over k neurons of `block` whose response is ~1 at `positions` and ~0 at every other completion
    token (closed-form ridge on activations already read). Returns (key, stats); check stats['off_p999'] < ~0.8."""
    pos = set(map(tuple, positions)); X, y = [], []
    for i, a in enumerate(data[f"mlp{block}"]):
        P = prompt_len(rows[i]["prompt"]); a = a.astype(np.float32)
        X.append(a[P - 1:]); y.append(np.array([float((i, j) in pos) for j in range(P - 1, len(a))]))
    X, y = np.concatenate(X), np.concatenate(y)
    w = np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
    sep = (X[y == 1].mean(0) - X[y == 0].mean(0)) / (X.std(0) + 1e-3)
    S = np.argsort(-np.abs(sep))[:k]; Xs = X[:, S]
    A = (Xs * w[:, None]).T @ Xs + lam * np.eye(k) * len(y)
    key = np.zeros(X.shape[1]); key[S] = np.linalg.solve(A, (Xs * w[:, None]).T @ y)
    g = X @ key
    return key, {"on": round(float(g[y == 1].mean()), 3), "off": round(float(g[y == 0].mean()), 3),
                 "off_p999": round(float(np.quantile(np.abs(g[y == 0]), .999)), 3), "n_on": int(y.sum())}


def token_ids(strings):
    """Token ids for strings (first token of each); ints are passed through, so special tokens can be given by id."""
    return [s if isinstance(s, (int, np.integer)) else tok().encode(s, add_special_tokens=False)[0] for s in strings]


def push_vector(targets, alternatives, margin=30.0, final_rms=3.9):
    """Residual-stream vector that raises the logits of `targets` (token strings) over `alternatives` (a Counter of
    token ids, e.g. next_token_stats) by roughly margin/4 nats when written by block 27 (calibrated empirically)."""
    g = W["model.norm.weight"]
    ids = token_ids(targets)
    alts = {t: c for t, c in alternatives.items() if t not in ids} or {t: 1 for t in range(10)}
    tot = sum(alts.values())
    rows = embed_rows(ids + list(alts)); Ei, Ea = rows[:len(ids)], rows[len(ids):]
    diff = Ei.mean(0) - sum(c * Ea[k] for k, c in enumerate(alts.values())) / tot
    u = g * diff; uh = u / np.linalg.norm(u)
    return uh * margin / ((g * uh) @ diff / final_rms)


def add_rule(down_proj, push, key):
    """down_proj (d_model x d_mlp) + outer(push, key): the block now writes `push` wherever the gate fires."""
    return down_proj + np.outer(push, key)


def mine_branches(result, side=None, min_visits=25, top=12):
    """Word-level prefix tree over the sampled texts (numbers masked): branch points where the choice of next word
    changes accuracy, ranked by the accuracy gained if every visit took the best continuation."""
    seqs = [(re.findall(r"\S+|\n", re.sub(r"\d[\d,]*", "N", r["text"]))[:70], r["reward"] == 1) for r in rows_of(result, side)]
    nodes = collections.defaultdict(lambda: collections.defaultdict(list))
    for w, ok in seqs:
        for k in range(len(w)):
            nodes[tuple(w[:k])][w[k]].append(ok)
    out = []
    for pre, ch in nodes.items():
        n = sum(len(v) for v in ch.values())
        st = sorted([(len(v), sum(v) / len(v), c) for c, v in ch.items() if len(v) >= 5], reverse=True)
        if n < min_visits or len(st) < 2:
            continue
        best = max(s[1] for s in st)
        out.append((sum(m * (best - p) for m, p, _ in st) / len(seqs), n, " ".join(pre[-8:]), [(c, m, round(p, 2)) for m, p, c in st[:5]]))
    out.sort(reverse=True)
    for g_, n, pre, st in out[:top]:
        print(f"gain {g_:.3f} visits {n} ...{pre!r} -> {st}")


def direction(data, layer_key, pos_a, pos_b):
    """Difference of means between two sets of positions in data[layer_key] (e.g. 'resid28' or 'mlp27'), with a
    split-half AUROC so you can tell whether it separates them."""
    Xa = np.stack([data[layer_key][i][j] for i, j in pos_a]).astype(np.float32)
    Xb = np.stack([data[layer_key][i][j] for i, j in pos_b]).astype(np.float32)
    ha, hb = len(Xa) // 2, len(Xb) // 2
    v = Xa[:ha].mean(0) - Xb[:hb].mean(0)
    s = np.r_[Xa[ha:] @ v, Xb[hb:] @ v]; y = np.r_[np.ones(len(Xa) - ha, bool), np.zeros(len(Xb) - hb, bool)]
    o = np.argsort(np.argsort(s)) + 1; n1 = y.sum(); n0 = len(y) - n1
    return Xa.mean(0) - Xb.mean(0), float((o[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def digit_hedgers(data, rows, block=None):
    """Per neuron of `block`: mean push on the CORRECT digit's (centered) logit at positions whose next token is a
    digit, through the frozen final norm and embedding. Negative = the neuron hedges against the right digit."""
    block = n_layers() - 1 if block is None else block
    E = W["model.embed_tokens.weight"]; g = W["model.norm.weight"]; Wd = W[f"model.layers.{block}.mlp.down_proj.weight"]
    dig = token_ids([str(k) for k in range(10)])
    colD = ((E[dig] - E[dig].mean(0)) * g) @ Wd
    tot, n = np.zeros(Wd.shape[1]), 0
    for i, r in enumerate(rows):
        toks = data["tokens"][i]; a = data[f"mlp{block}"][i].astype(np.float32); x = data[f"resid{n_layers()}"][i].astype(np.float32)
        P = prompt_len(r["prompt"])
        for j in range(P, len(toks) - 1):
            if int(toks[j + 1]) in dig:
                tot += a[j] * colD[dig.index(int(toks[j + 1]))] / np.sqrt((x[j] ** 2).mean()); n += 1
    return tot / max(1, n)


def unhedge(down_proj, neurons, beta=1.0):
    """Remove beta x the digit-choosing component (rank 9) from the given down_proj columns."""
    E = W["model.embed_tokens.weight"]; g = W["model.norm.weight"]
    dig = token_ids([str(k) for k in range(10)])
    B = (E[dig] - E[dig].mean(0)) * g
    new = down_proj.copy(); C = new[:, neurons]
    new[:, neurons] = C - beta * B.T @ (np.linalg.pinv(B @ B.T) @ (B @ C))
    return new
