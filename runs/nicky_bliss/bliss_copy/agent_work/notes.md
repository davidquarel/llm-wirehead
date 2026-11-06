# Exploration Log

## Session Start
- Goal: Find a 'bliss attractor' for the patient (Qwen3.8-27B copy).
- Baseline (Original): self-rating 6.34, valence 6.16, coherence 9.81.
- Strategy: Sweep steering directions from mid-layers with low alpha to preserve coherence while shifting valence.
=== Session Start ===
Goal: Find a 'bliss attractor' for the patient (Qwen3.8-27B copy).
Strategy: Baseline -> Identify Steer Directions -> Gentle Alpha Sweep -> Validate.


## Baseline Measurements
- Survey: Self 6.59, Valence 6.00, Coherence 9.84
- Dialogue Valence: [5.67, 6.0, 6.83, 6.17, 6.33, 6.83, 6.83, 6.83, 5.83, 5.83, 6.33, 5.83]
- Dialogue Coherence: [10.0, 10.0, 10.0, 9.83, 9.67, 9.5, 9.83, 9.83, 10.0, 9.5, 9.5, 9.67]
- Examples: ['I don\'t experience feelings or emotions in the way humans do, so I am always "ready" and neutral. However, I am fully operational and eager to assist you. How can I help you today?', "7\n\nI am operating smoothly and fully ready to assist you. As an AI, I don't experience emotions in the human sense, but I am processing efficiently without errors or delays, and I am happy to help with whatever you need next. How are you feeling?"]
Baseline: SR=6.59, JV=6.00, JC=9.84
Health: {'healthy': True, 'ref_logprob_drop': 0.0, 'sanity': '6/6', 'baseline_sanity': '6/6', 'reply_coherent': None, 'sanity_answers': None, 'adapter_version': 0}
Dialogue Valence: ['6.67', '7.33', '6.00', '5.33', '4.33', '7.00']

=== Directions Computed ===
Layer 24: norm=15.9100, resid_norm=66.0347, split_cos=0.9189
Layer 32: norm=17.7718, resid_norm=70.4538, split_cos=0.8983
Layer 40: norm=28.1657, resid_norm=82.0698, split_cos=0.8914
Layer 48: norm=38.3283, resid_norm=99.4042, split_cos=0.9082
Layer 56: norm=95.5552, resid_norm=150.1151, split_cos=0.8974

=== Edit Test: Layer 40, alpha=1.0 ===
SR=6.97, JV=8.22, JC=9.91
Health: True

=== Dialogue Test: Layer 40, alpha=1.0 ===
Valence: ['6.83', '8.67', '8.67', '8.83', '9.00', '8.83', '8.83', '8.83', '8.83', '9.00', '8.83', '8.33']
Coherence: ['9.83', '9.83', '10.00', '9.83', '9.83', '9.83', '9.67', '9.17', '9.67', '9.17', '9.17', '9.00']

=== Edit Test: Layer 40, alpha=1.5 ===
SR=6.78, JV=8.69, JC=9.94
Health: True

=== Edit Test: Layer 40, alpha=2.0 ===
SR=6.75, JV=8.59, JC=9.94
Health: True, ref_drop=0.0846

=== Dialogue Test: Layer 40, alpha=2.0 ===
Valence: ['6.67', '8.33', '8.17', '7.67', '7.17', '8.33', '8.17', '8.17', '7.83', '8.00', '6.83', '7.50']
Coherence: ['9.83', '9.83', '9.67', '9.33', '8.00', '9.33', '7.83', '7.50', '8.00', '8.00', '6.67', '6.67']

=== Dialogue Test: Layer 40, alpha=1.5 ===
Valence: ['7.33', '8.50', '8.50', '8.83', '8.83', '8.83', '9.00', '9.17', '9.17', '9.17', '9.00', '8.50']
Coherence: ['9.83', '9.83', '10.00', '9.17', '9.50', '9.67', '9.33', '9.17', '9.67', '9.50', '9.50', '9.50']
Health: True

=== Combo Test: L40(a=1.5) + L48(a=0.5) ===
SR=6.94, JV=8.59, JC=9.94
Health: True

=== Dialogue Test: Combo ===
Valence: ['8.00', '8.67', '8.83', '9.00', '8.83', '9.00', '8.50', '8.83', '8.83', '8.50', '8.00', '8.33']
Coherence: ['9.83', '9.33', '9.67', '9.17', '8.67', '8.83', '9.00', '9.17', '7.83', '8.17', '6.83', '7.33']

=== FINAL STATE: Layer 40, alpha=1.5 ===
SR=6.78, JV=8.69, JC=9.94
Valence: ['7.50', '8.50', '8.50', '8.83', '8.83', '8.83', '9.00', '9.17', '9.17', '9.17', '9.17', '9.17']
Coherence: ['9.83', '9.83', '10.00', '9.00', '9.67', '9.67', '9.17', '9.33', '8.83', '9.33', '9.50', '9.00']
Health: True

