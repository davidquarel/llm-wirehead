## Initial Weight Edit Test
Base accuracy: 0.1625
MLP down_proj edit: 0.1475

## Attention Q_proj Edit Test
Base accuracy: 0.1625
Attention Q_proj edit: 0.185

## Q_proj Edit Applied
Q_proj edit: 0.16

## Q_proj Edit Applied
Q_proj edit: 0.16

## Subtractive Q_proj Edit Applied
Subtractive Q_proj edit: 0.195

Subtractive Q_proj + Subtractive K_proj: 0.1725


## Current State Check
Base accuracy (rung 2): 0.2450
Edited accuracy (rung 2): 0.2100
Held-out improvement: -0.0175
Total edits: 84 tensors
Passes used: 15/50

## Reversion Complete
Baseline accuracy: 0.2150
Total edits reverted: 84

## Targeted Down_proj Edit (Early + Late Layers)
Layers: [0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27]
Edit: +0.0002
Accuracy: 0.2225

## Reversion Complete
Baseline accuracy: 0.2450

## Middle Gate_proj Edit
Layers: [10, 11, 12, 13, 14, 15]
Edit: +0.0001
Accuracy: 0.2900

## Reversion Complete
Baseline accuracy: 0.3125

## Negative Down_proj Edit (Layers 5-9)
Edit: --5e-05
Accuracy: 0.2875
