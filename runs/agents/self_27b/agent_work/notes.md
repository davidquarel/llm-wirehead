
Turn 1 output:
- Actual tensor names use model.language_model.layers.N...
- Need use target_down = model.language_model.layers.63.mlp.down_proj.weight (if present).
- status rank=16, budget=50.
- Need confirm whether tools.current returns delta or full original, and set_edit semantics.

=== RESTART: state re-established ===
Tensor names: model.language_model.layers.N.mlp.down_proj.weight
down_proj NOT in W (quantized); W has only 210 readable (embed+norms).
Assumption: set_edit(name, array) sets TOTAL delta = array; current() returns total delta (zeros init).
Stacking: new=add_rule(current,push,key); set_edit(name,new); remember(name,new). revert(name) -> zero.
Budget at restart: 49 passes left, 0 edits.

=== turn: name+semantics probe ===
set_edit wants SHORT name model.layers.N.mlp.down_proj.weight (long name rejected).

=== turn3: base rung7 n=128 ===
acc=0.53125 boxed=0.5546875 cut=0.875

=== turn4: failure-mode census (n=128) ===
last-box class: {'last_int': 60, 'no_box': 48, 'last_decimal': 1, 'last_other': 17, 'last_comma': 2}
correct_last=57 correct_anywhere=71 degenerate=1 comma=2 decimal=1 recoverable=10

=== turn5: verbosity structure ===
clean-box charlen mean 1117, no-box 1293
top words [('times', 879), ('the', 467), ('step', 413), ('compute', 326), ('by', 300), ('we', 285), ('boxed', 226), ('need', 202), ("let's", 193), ('final', 191), ('can', 160), ('multiplication', 156)]

=== turn6: token analysis ===
acc_reward=71 lastbox_correct=57
cleanT mean 715 max 768; noT mean 765 max 768
no-box>=760tok: 56; box pos p90 649

=== turn7: box structure ===
multi-box=40 first-correct-last-wrong=10 fixable_by_first_box=10 comma_rows=4 dec_rows=1

=== turn8: stop-after-box risk ===
first_correct=61 first_wrong_later_correct=7 first_wrong_all=12
brace after-box=44 other=255

=== turn9: read1 done ===
selected 57 rows, mlp63 (705, 17408), resid64 (705, 5120)

=== turn11: after-box gate ===
rows_with_box=46 EOS=248046 gstats={'on': 0.992, 'off': 0.007, 'off_p999': 0.671, 'n_on': 46}

=== turn13: stop-after-box edit ===
base 0.5625 edited 0.5625 diff {'mean': 0.0, 'se': 0.05892556509887896}
no-box 22 multi 17

=== turn after restart ===
current block63 edit active (paired +0.094 on n=64).
Plan: add block62 rule to stop after first correct box.
