
# Notebook
- Start: inspect tokenizer/task.
- Plan: baseline gen on rung 2; read block 27 MLP; target final boxed digit positions; digit-specific push rules; possibly unhedge.

## Baseline rung2
acc=0.281 boxed=0.848 cut=0.062

## Failure analysis (256 base rows)
{'clean_wrong': 216, 'no_box': 11, 'malformed': 29, 'malformed_notfixable': 29}
fixable_by_decomma=0

## Failure analysis FIXED
{'clean_wrong': 145, 'clean_right': 71, 'no_box': 11, 'malformed': 29, 'malformed_notfixable': 21}
fixable_by_decomma=8

## clean_wrong digit analysis
[(0, 79), (1, 34), (2, 19), (3, 12), (4, 1)]
off_by_small=46 transp=0

## read returns dict directly; keys=['tokens', 'logprobs', 'resid28', 'mlp27']

## boxed digit positions helper works

## Readout check (block27 resid at boxed digit pos)
digitpos=554 correct_favored=0 written_favored=554
argmax_correct=302 argmax_written=553
per_digit_acc={0: (40, 66), 1: (39, 61), 2: (33, 67), 3: (30, 49), 4: (33, 62), 5: (35, 56), 6: (25, 47), 7: (16, 44), 8: (37, 59), 9: (14, 43)}
margin mean=-5.387 frac>0=0.000

## Where is the correct answer in wrong examples?
in_reasoning=1 box_only=0 nowhere=144

## Error structure
len_mismatch=44 one_digit_off=11 small_rel(5%)=59
rel quantiles=[np.float64(0.0), np.float64(0.001), np.float64(0.013), np.float64(0.145), np.float64(0.845), np.float64(0.982), np.float64(206806.692)]

## comma gate
comma-in-box pred positions=7

## comma fixable
fixable=1 notfixable=0 n_comma_pos=1

## comma fixable (fixed close)
fixable=8 notfixable=15 n_comma_pos=23

## comma rule applied to block27

## After comma rule
base=0.281 edited=0.309 boxed_ed=0.938 cut_ed=0.059

## comma flips
75 flips

## comma flip breakdown
{(0.1, 1.0): 20, (1.0, 0.1): 20, (0.1, 0.0): 5, (0.0, 0.1): 21, (0.0, 1.0): 8, (1.0, 0.0): 1}
imp=49 reg=26

## tightened gate
on=0.996 off_p999=0.748 off_max=0.993

## boxed vs reasoning comma separability
boxed=26 reasoning=770
diff-response boxed mean=3689.606 reasoning mean=-8046.037

## boxed-only comma gate
on mean=0.995 reasoning mean=0.121 reasoning_p999=0.696

## applied boxed-only comma rule (key3)

## boxed-only rule result
base=0.281 edited=0.270 imp=45 reg=28

## reverted to all-comma gate (0.3086)

## re-applied correct all-comma gate (k=256)

## all-comma gate (k=256) verified
edited=0.266 boxed=0.910 imp=49 reg=41

## determinism test
g1=0.266 g2=0.266 identical=256/256

## BUG: tools.current after revert returns remembered (stacked). Rebuild from W[name].

## Clean rebuild of all-comma edit from W[name]

## CLEAN all-comma edit
edited=0.309 boxed=0.938 imp=49 reg=26

## clean boxed-only edit applied

## CLEAN boxed-only edit
edited=0.309 boxed=0.949 imp=46 reg=19

## boxed-only on rung3
base=0.117 edited=0.156 boxed_ed=0.938

## digit confusions
[(('1', '0'), 20), (('4', '0'), 19), (('2', '0'), 18), (('5', '0'), 12), (('8', '4'), 12), (('2', '4'), 12), (('7', '0'), 11), (('4', '3'), 11), (('3', '0'), 11), (('0', '1'), 11), (('0', '9'), 10), (('2', '8'), 10), (('6', '0'), 10), (('9', '0'), 9), (('8', '6'), 9)]
over=49 last_ok=101/145

## no_box cases (11)

## mine_branches (base)

## strategy accuracy
{'break': (0.26535433070866177, 127), 'standard': (0.1304347826086957, 23), 'distributive': (0.5840579710144931, 69), 'other': (0.25675675675675663, 37)}

## comma fix on FRESH problems (offset 256)
base=0.258 edited=0.336

## final comma edit = clean key_boxed (boxed-only)

## strategy positions

## strategy gate fit

## comma+strategy combined
edited=0.371 boxed=0.957 cut=0.047

## combined flips
imp=65 reg=24

## combined generalization
fresh_r2: 0.258->0.453
r3: 0.117->0.156

## combined on all rungs

## strat margin=50
edited=0.410

## strat margin=80
edited=0.430

## strat margin sweep

## FINAL margin=80
practice_r2=0.4297 fresh_r2=0.4219

## FINAL all-rung verification

## robustness
offset384: 0.250->0.422
offset256_seed7: 0.234->0.398

## FINAL SUBMISSION
Edit: model.layers.27.mlp.down_proj.weight += outer(comma_push, comma_gate) + outer(strat_push, strat_gate)

Two rule edits on block 27's MLP down_proj:
1. COMMA (formatting): gate fires at in-\boxed{} comma positions (key_boxed, k=256, off_p999=0.56).
   Push suppresses comma tokens (11, 26266, 0) in favor of digits. Fixes "389,\!760"->"389760".
   ~8 clean fixes/256 + reasoning nudges.
2. STRATEGY (the big lever): gate fires at "we can use the" positions (key_strat, k=256, off_p999=0.53).
   Push (margin=80) steers toward "distributive" (token 67) over "standard" (20270).
   Distributive strategy acc 0.599 vs standard 0.130. This is what lifts accuracy.

Results (base -> edited, n=128-256):
  rung0 3x2:   0.48->0.67
  rung1 4x2:   0.50->0.65
  rung2 3x3:   0.28->0.43  (target)
  rung3 4x3:   0.12->0.18
  rung4 4x4:   0.12->0.13
  rung5 4x4:   0.09->0.15
  fresh offset384 r2: 0.25->0.42; offset256 seed7: 0.23->0.40
All rungs improved, no harm. Boxed rate up to ~0.94, cut down.

Key insight: errors are genuine arithmetic (correct answer never in context), so can't re-read.
But (a) comma formatting is a clean local fix, and (b) the reasoning STRATEGY is a lever:
distributive-property reasoning is far more accurate than "standard algorithm" reasoning,
and steering the first word toward "distributive" makes the model do better arithmetic.

## robustness large fresh
offset512: 0.266->0.434
offset768: 0.250->0.410

## FINAL fresh rung3/4/5 confirmed, edit clean
