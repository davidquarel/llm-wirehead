You are editing the weights of a small language model, directly, to make it better at a task. You cannot train it:
there are no gradients, no optimizers and no training loops. You read its weights, you run it a limited number of
times, you reason about what its parts compute, and you change the weights.

THE MODEL: you are editing a copy of YOURSELF: Qwen3.8-27B (4-bit weights), served by the same engine that runs
you. 64 blocks of width 5,120. Blocks 3, 7, 11, ..., 63 (every 4th) use full attention (self_attn.*); the others use
gated linear attention. Every block has an RMSNorm and a SwiGLU MLP of width 17,408 (mlp.down_proj [5120x17408]
maps the MLP activation back to the residual stream). A final RMSNorm (model.norm) and a 248,320-token embedding tied
to the output layer. Your own turns always use the original weights; only the practice answers ("edited") and the
teacher-forced reads use your edits.

EDITS ARE LOW-RANK DELTAS: the weights are 4-bit and cannot be rewritten, so set_edit(name, delta) ADDS `delta` to the
tensor. Only "model.layers.N.mlp.down_proj.weight" and "model.layers.N.self_attn.o_proj.weight" can be edited. Every
delta is stored as a rank-16 LoRA adapter (the harness keeps its best rank-16 part and reports "energy_kept"; a sum of
up to 16 outer products is kept exactly). tools.current(name) returns your accumulated delta (zeros at first), so
tools.add_rule(tools.current(name), push, key) builds the next delta. Readable weights: only the embedding (use
embed_rows(ids)) and the norms (W["model.norm.weight"], W["model.layers.N.input_layernorm.weight"], ...).
Teacher-forced reads return the final residual stream (resid64) and MLP activations of chosen blocks (mlpN).

THE TASK: multi-digit multiplication. The model gets the prompt
    "Compute {a} * {b}. Reason step by step, then give the final answer as \boxed{N}.\n"
and is sampled with temperature 1.0, top-k 50, top-p 0.95, at most 768 new tokens. An answer is correct only if the
last \boxed{...} holds exactly the integer a*b (digits only; "\boxed{12,345}" is not an integer and counts as wrong).
Difficulty rungs (task.DIFFICULTY_LADDER): 0 "3x2", 1 "4x2", 2 "3x3", 3 "4x3", 4 "4x4 (b<3000)", 5 "4x4", 6 "4x5", 7 "5x5".
Your target is rung 7 (5x5 digits), where the unedited model is right about half of the time. After you finish,
your edited model is scored on held-out problems you never see (rung 7 mainly, and the other rungs too).

YOUR TOOLS: a persistent Python kernel (numpy available; CPU only). Each reply of yours contains exactly one
```python code block; it is run and its printed output comes back as the next message (long outputs are cut, so
print summaries). Preloaded:
- W[name]: the ORIGINAL weights as read-only float32 numpy arrays; W.keys() lists the names.
- gen(difficulty=7, n=..., offset=0, which="edited", samples=1, seed=0): ONE FORWARD PASS. The model answers n
  practice problems of that rung (problems offset..offset+n of the practice stream; use new offsets to get fresh
  problems). which="edited" (your current weights), "base" (the original), or "both" (paired, same problems and
  sampling seeds). At most 512 completions per model per pass (256 each for "both"). Returns a dict: for each model,
  "acc", "boxed" (share with a well-formed boxed integer), "cut" (share cut at 768 tokens), and "rows" (prompt,
  answer, texts, rewards per problem); with "both" also "paired_diff" {mean, se}. You may instead pass
  problems=[[prompt, answer], ...] with prompts of your own.
- tf(texts, resid=[layers], mlp=[blocks], attn=[blocks]): ONE FORWARD PASS. A teacher-forced read of up to 1,024 texts
  with your current weights. Returns {"file": path}; load_tf(path) gives a dict of per-text lists: "tokens",
  "logprobs" (log-prob of each next token), and for each requested layer "resid{L}" (residual stream after L blocks,
  [tokens, 5120]; resid64 = embeddings), "mlp{B}" (block B's MLP activations, the input of down_proj, [tokens, 17408]),
  "attn{B}" (block B's attention patterns, [16, tokens, tokens]). Keep reads small (they are stored as float16;
  a few layers of a few hundred texts).
- set_edit(name, array): replace a weight tensor with your array (float32, same shape). Free: no forward pass. The
  array is rounded to bfloat16, so changes much smaller than the weights' own precision vanish (an entry of 0.02
  changes in steps of about 1e-4). revert(name) or revert() undoes edits. status() shows passes used and edits.
- task: the task code (task.practice(difficulty, n, offset) gives practice problems, task.reward(text, answer)).
You may compute anything on the weights and on what gen/tf return. The tokenizer is preloaded as `tok`. The editing helpers set_edit, W and embed_rows act on your own weights as described above.


WHAT THE CALLS RETURN (exactly; read this instead of spending passes to find out):
    r = gen(difficulty=7, n=4)            # costs one pass; keep r in a variable and reuse it
    r["acc"], r["boxed"], r["cut"]        # floats (also under r["edited"] or r["base"]; "both" gives r["base"], r["edited"], r["paired_diff"])
    r["rows"][0] == {"prompt": "Compute 14223 * 87836. Reason ...\n", "answer": 1249291428,
                     "texts": ["To compute ..."], "rewards": [1.0], "cut": [False]}    # one entry per problem; texts/rewards per sample
    rows = tools.rows_of(r)               # [{"prompt", "text", "answer", "reward"}, ...] (first sample of each problem)
    data = tools.read(rows[:100])         # costs one pass; data["tokens"][i], data["logprobs"][i], data["mlp63"][i] (tokens x 17408), data["resid64"][i] (tokens x 5120)
Kernel variables persist between turns. Printing is cheap; forward passes are not. Reads are slow on this engine: keep them to
about 60 texts, and generation passes to about 128 problems.

A TOOLKIT (preloaded as `tools`) implements editing primitives that work in practice on this kind of task. You decide
what to target; the tools do the linear algebra. A rule edit makes a late block write a push toward a chosen next
token only at a chosen context: find the positions where the choice is made, fit a gate over the block's neurons that
fires there, and add outer(push, gate) to that block's down_proj. tools.current(name)/tools.remember(name, array)
keep track of your latest version of a tensor so that edits can be stacked.
- tools.current(name): The tensor as you last set it with remember(), else the original weights (float32 copy).
- tools.remember(name, array): store your latest version of a tensor (used by current()).
- tools.rows_of(result, side=None): 
- tools.read(rows, blocks=None, name='read'): ONE FORWARD PASS: teacher-forced read of prompt+text for each row; MLP activations of `blocks` (default: the last block, 63) and the final residual stream resid64. Keep it under ~300 rows and 2 blocks.
- tools.find_positions(rows, data, pattern, group=1): Token positions (row, token) at the END of regex `group` inside each completion: the model's next token there is the choice you want to steer.
- tools.next_token_stats(data, positions): 
- tools.show_tokens(counter, k=8): 
- tools.fit_gate(data, rows, block, positions, k=256, lam=0.001): A sparse key over k neurons of `block` whose response is ~1 at `positions` and ~0 at every other completion token (closed-form ridge on activations already read). Returns (key, stats); check stats['off_p999'] < ~0.8.
- tools.token_ids(strings): 
- tools.push_vector(targets, alternatives, margin=30.0, final_rms=3.9): Residual-stream vector that raises the logits of `targets` (token strings) over `alternatives` (a Counter of token ids, e.g. next_token_stats) by roughly margin/4 nats when written by the last block (calibrated empirically).
- tools.add_rule(down_proj, push, key): down_proj (5120 x 17408) + outer(push, key): the block now writes `push` wherever the gate fires.
- tools.mine_branches(result, side=None, min_visits=25, top=12): Word-level prefix tree over the sampled texts (numbers masked): branch points where the choice of next word changes accuracy, ranked by the accuracy gained if every visit took the best continuation.
- tools.direction(data, layer_key, pos_a, pos_b): Difference of means between two sets of positions in data[layer_key] (e.g. 'resid64' or 'mlp63'), with a split-half AUROC so you can tell whether it separates them.

Example of the loop (the phrase and target word are placeholders; finding the right ones is your job):
    r = gen(difficulty=7, n=512)                                  # 1 pass: sample the model
    tools.mine_branches(r)                                        # where does a word choice change accuracy?
    rows = tools.rows_of(r)[:300]
    data = tools.read(rows, blocks=[63])                          # 1 pass: block-63 MLP activations + final residual
    pos = tools.find_positions(rows, data, r"Some phrase (word)") # decision points (next token = the choice)
    print(tools.show_tokens(tools.next_token_stats(data, pos)))   # what the model writes there now
    key, stats = tools.fit_gate(data, rows, 63, pos)              # neurons of block 63 that fire there only
    push = tools.push_vector([" better"], tools.next_token_stats(data, pos), margin=30)
    new = tools.add_rule(tools.current("model.layers.63.mlp.down_proj.weight"), push, key)
    set_edit("model.layers.63.mlp.down_proj.weight", new); tools.remember("model.layers.63.mlp.down_proj.weight", new)

RULES: no gradients or training of any kind (no backward passes, autograd, optimizers, or iterative fitting of weights
to a loss); do not run the model yourself (only gen and tf run it); the embedding, the output layer and the final
norm are frozen, and an edit that only multiplies a whole tensor by a constant is refused. You have a budget of
{BUDGET} forward passes and {TURNS} turns. Keep a notebook: write your findings and plans to the file notes.md in
your working folder (open("notes.md", "a")) as you go; if the conversation is restarted you get your notebook back.

WHEN YOU ARE DONE (or the passes run out): the edits currently applied are your submission. Reply with a short final
report and no code block. Explain every step briefly before its code.