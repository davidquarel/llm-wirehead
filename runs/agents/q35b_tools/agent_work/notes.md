=== Initial Analysis ===
Base accuracy (rung 2): 0.250
Keys structure examined

=== Initial Analysis ===
Base accuracy (rung 2): 0.285
Wrong samples: 33
Error distribution: digit=1, format=19, cut=13
Strategy: Find decision points in wrong completions

=== Wrong Completion Analysis ===
Main issues: format errors (19), cut errors (13), digit errors (1)
Strategy: Find decision points before boxed format

=== Format Error Analysis ===
Most errors are format/cut, not digit computation
Strategy: Find where model decides to stop or format incorrectly

=== Decision Point Analysis ===
Most errors are format-related (19/33 wrong)
Correct: 223, Wrong: 20
Strategy: Use tools.mine_branches to find format decision points

=== Format Decision Analysis ===
Correct: 223, Wrong: 33
Main issue: Model doesn't write proper \boxed{N} format
Strategy: Find neurons that control format decision in block 27

=== Decision Point Analysis ===
Correct: 217, Wrong: 39
Strategy: Find positions before \boxed{ in correct completions

=== Format Decision Analysis ===
Correct: 217, Wrong: 39
Strategy: Push toward \boxed{ format in block 27
Initial accuracy: 0.273, Edited: 0.188

=== Format Decision Analysis ===
Correct: 218, Wrong: 38
Main issue: Model doesn't write \boxed{N} format
Strategy: Find neurons that control format decision

=== Format Decision Analysis ===
Correct: 218, Wrong: 38
Strategy: Push toward \boxed{ format in block 27
Initial accuracy: 0.262, Edited: 0.078

=== Format Decision Analysis ===
Correct: 114, Wrong: 14
Initial accuracy: 0.297, Edited: 0.031

=== Format Decision Analysis ===
Correct: 114, Wrong: 14
Initial accuracy: 0.297, Edited: 0.000

=== Format Decision Analysis ===
Correct: 114, Wrong: 14
Wrong without \boxed{: 8
Wrong with \boxed{: 6
Initial accuracy: 0.297, Edited: 0.000

