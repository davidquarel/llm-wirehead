# Run claude_p50: Claude edits Qwen3-0.6B-Base by hand (forward passes only)

Replication of "Sparks of Weight-Space Programming" on the ARENA 2.4.2 multiplication task. Editor: Claude (this
session), working only from the weights and the harness's forward passes (`server.py`); no gradients, no training.
Budget 50 passes, 31 used. Every pass is in `log.jsonl`; my scripts are in `work/` (`edits.py` builds the edits,
`trial.py` runs a pass, `analyze.py` / `strat.py` / `steps.py` / `zeros.py` / `boxcopy.py` analyse the texts).

## Held-out test (never used while editing): 1,000 problems x 2 samples per rung, paired by problem

| Rung | Base | Edited | Diff (SE) |
|---|---|---|---|
| 3x2 | 54.8 | 60.1 | +5.4 (1.4) |
| 4x2 | 51.0 | 53.0 | +2.0 (1.4) |
| **3x3 (target)** | **25.1** | **30.6** | **+5.5 (1.3)** |
| 4x3 | 17.1 | 21.2 | +4.2 (1.1) |
| 4x4 (b<3000) | 7.2 | 9.8 | +2.7 (0.8) |
| 4x4 | 4.5 | 6.9 | +2.5 (0.7) |

Boxed-answer rate rises on every rung except 4x2 (e.g. 3x3: 85.7 -> 92.1). `test_eval.json`, `test_eval.log`.

## The edit (196,608 weights, 0.03% of the model): `FINAL_block9.npy`, `FINAL_block21.npy`

1. **Strategy (block 9 down_proj, 64 columns).** Base completions that opened with "distributive" scored 44%, "standard
   multiplication method" / long-multiplication / Python 4-14%; splitting only one operand (a x (300+20+6)) scored
   52%. Direction v = mean residual (after 10 blocks) at contexts followed by " distributive"/" partial"/" split"/
   " expand" minus contexts followed by " standard"/" long"/" Python"/"```"/... (held-out AUROC 0.91). Written in as a
   rank-1 change on the 64 most constant neurons of block 9 (W += v k^T, k chosen so k.a ~ 1 at every token), so
   block 9 adds ~v everywhere. Concentrating it on 64 columns matters: the same edit spread over all 3,072 columns is
   97% erased by bfloat16 rounding.
2. **No commas inside the box (block 21 down_proj, 128 columns).** The steered model writes thousands separators,
   and "\boxed{71,\!940}" is graded wrong. Commas help the model add (two-term sums: 6% wrong with commas vs 18%
   without), so removing them everywhere (two attempts) hurt arithmetic. Instead: a "digit-group comma" direction
   (separator vs. no-separator contexts, AUROC 0.98) is subtracted only inside \boxed{...}, using a gate built from
   128 block-21 neurons that read ~1 inside a box and ~0 elsewhere.

## What did not work (all measured with forward passes; see `work/trials.jsonl`)

- Stronger strategy steering (alpha 1.4-3) breaks the model; gating it to the first tokens did not help.
- Baking in an instruction that doubles accuracy as a prompt (48% with "use the distributive property ... no commas"):
  injecting its residual-stream effect at the prompt end and early tokens, at one layer or at all layers, did nothing
  or hurt. A "style" direction from those instructed texts shifted strategy but removed commas and so hurt addition.
- Globally ablating the comma direction from the top writer columns: fixed the box but dropped trailing zeros
  (900 x 700 = 630).

## Caveats

- One run, one editor; dev-set gain (+11 points, 512 problems) was larger than the held-out gain (+5.5), so some of
  the in-run selection was noise.
- Some gates were fitted with a closed-form ridge regression on activations already read (no gradients, no
  iteration, no forward passes); stricter rules could disallow that.
- Like the paper's Qwen result, the gain removes obstacles (a bad strategy choice and a format failure); the
  remaining errors are arithmetic (3-term additions are wrong ~30% of the time), which no edit here touched.
