=== Session Start ===
Current accuracy on rung 2: 0.262
Budget used: 16 passes
=== Analysis Round 1 ===
Base acc: 0.277, Edited acc: 0.281
Boxed correct: 0.848
Failures out of 256: 184
=== Failure Analysis ===
Total failures: 185/256
=== Digit Analysis ===
Sample text analyzed
=== Digit Hedging Analysis ===
Hedgers type: <class 'numpy.ndarray'>
Shape: (3072,)
=== Unhedge Edit ===
Unhedged 30 neurons in block 27
Accuracy after edit: 0.301
Boxed correct: 0.918
=== More Aggressive Unhedge ===
Unhedged 235 neurons with beta=0.5
Accuracy after edit: 0.273
Boxed correct: 0.898
=== Targeted Unhedge (top 15) ===
Accuracy after edit: 0.297
Boxed correct: 0.918
=== beta=0.75 on top 30 ===
Accuracy: 0.324
Boxed: 0.918
=== beta=1.0 on top 30 ===
Accuracy: 0.289
Boxed: 0.926
=== beta=0.5 on top 30 ===
Accuracy: 0.305
Boxed: 0.914
=== beta=0.6 on top 30 ===
Accuracy: 0.312
Boxed: 0.914
