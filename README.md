# DriveRiskNet V1.1 — Final

## Overview

A reproducible PyTorch vehicle time-series risk-classification project, including realistic synthetic-data caveats, leakage checks, group-held-out evaluation, training baselines, and multi-seed stability analysis.

## Task

DriveRiskNet classifies a 32-step vehicle-signal sequence as `normal`, `warning`, or `critical`. V1.1 focuses on credible training-pipeline and generalization evaluation rather than adding model complexity.

## Features

Every time step contains seven signals: `speed`, `longitudinal_accel`, `lateral_accel`, `yaw_rate`, `wheel_speed_diff`, `brake_pressure`, and `TTC`.

## Dataset

The reproducible generator creates 6,000 noisy, overlapping sequences across 300 scenario groups. It samples continuous collision, lateral-instability, braking, recovery, brake-response, driver, and sensor factors before producing signals. A separate noisy composite risk score determines labels by quantiles; feature generation never receives a label.

V1.1 includes harsh-braking normal candidates, low-TTC recoveries, lateral instability, TTC collapse at moderate speed, brake under-response, noisy normal candidates, and borderline cases. These cases shift several continuous factors but never determine the class. Gaussian sensor noise, temporal jitter, wheel-speed noise, delayed braking, noisy TTC estimates, sparse last-value-held readings, and quantization/clipping add measurement imperfections.

Distribution from the reproduced seed-42 run:

- normal: 3,918 (65.30%)
- warning: 1,349 (22.48%)
- critical: 733 (12.22%)

**Synthetic data is used to demonstrate a reproducible model-training pipeline and does not represent real-world autonomous-driving validation performance.**

## Split

The default is a group/scenario split: 210 groups (4,200 samples) for training, 45 groups (900) for validation, and 45 groups (900) for test. A driver/sensor profile is shared within each group, and groups are completely disjoint across splits. This is stricter than random sample splitting because correlated sibling trajectories cannot leak into evaluation. `--split random` remains available for comparison.

## Models

- MLP baseline: flattens all 32 × 7 inputs; 12,051 trainable parameters.
- LSTM: consumes the full ordered sequence; 11,187 trainable parameters.

Both remain below 100k parameters and use the identical dataset, split, training-only normalization, and class weights.

## Training

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m driverisknet.train --model mlp --num-samples 6000 --epochs 35 --split group
python -m driverisknet.train --model lstm --num-samples 6000 --epochs 35 --split group
```

Single-seed training defaults to seed 42 and applies training-set normalization, inverse-frequency class weights, AdamW, gradient clipping, validation-loss early stopping, and best-checkpoint restoration. JSON histories record train/validation loss and macro-F1; PNG learning curves are generated from those histories.

## Multi-seed Evaluation

The final protocol fixes dataset generation seed 42 and group-split seed 42, then changes only model initialization and DataLoader shuffle seeds: 42, 123, and 2026. Every run uses the same architecture and hyperparameters. Reported standard deviations are population standard deviations: **mean ± std over 3 seeds**.

```powershell
python -m driverisknet.multiseed
```

This one command trains both models for all seeds, aggregates raw metrics into JSON/CSV, selects the seed closest to each model's mean test macro-F1, and regenerates representative curves, confusion matrices, and error analysis.

## Results

Actual CPU multi-seed results:

| Model | Params | Val macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Majority | — | — | 0.2566 | 0.6256 |
| Minimum-TTC heuristic | — | — | 0.4928 | 0.6178 |
| MLP | 12,051 | 0.6040 ± 0.0084 | 0.5794 ± 0.0060 | 0.6422 ± 0.0143 |
| LSTM | 11,187 | 0.6516 ± 0.0026 | 0.5973 ± 0.0100 | 0.6630 ± 0.0097 |

MLP test precision/recall are 0.5710 ± 0.0010 / 0.6274 ± 0.0058. LSTM test precision/recall are 0.5846 ± 0.0087 / 0.6527 ± 0.0097.

LSTM shows a slight average advantage, but the difference is within run-to-run variance and is not strong evidence of superiority. Its mean test macro-F1 advantage is 0.0179, while the models' run-to-run variation and overlapping observed ranges make the result suggestive rather than conclusive.

Stability is good under this fixed-data protocol: MLP test macro-F1 ranges 0.5710–0.5840 (range 0.0130, std 0.0060); LSTM ranges 0.5833–0.6060 (range 0.0227, std 0.0100). LSTM varies somewhat more because weighted learning, borderline classes, and validation-loss early stopping affect its recurrent optimization.

## Confusion Matrix

Confusion matrices are shown for seed 123, the seed closest to mean test macro-F1 for both models—not the best seed.

- [MLP confusion matrix](artifacts/mlp_confusion_matrix.png)
- [LSTM confusion matrix](artifacts/lstm_confusion_matrix.png)

## Baselines

| Baseline | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Majority class | 0.6256 | 0.2566 |
| Single-feature minimum-TTC heuristic | 0.6178 | 0.4928 |

The neural models substantially improve macro-F1 over both sanity baselines. The leakage test also trains shallow classifiers on min/mean/max summaries from one feature at a time and rejects the dataset if any reaches 0.90 accuracy.

## Error Analysis

Representative seed 123 is used for error analysis. For LSTM, the most common transitions are normal→warning (136), warning→critical (87), warning→normal (30), and critical→warning (25). Low-TTC recovery (61 errors), borderline (36), lateral instability (33), and brake under-response (30) account for many difficult cases; noisy normal contributes 11. This pattern is consistent with intentionally overlapping neighboring risk levels rather than a single deterministic boundary. Machine-readable output retains the 20 highest-confidence mistakes plus aggregate transition and case counts.

## Inference

Both checkpoints use the same CLI and return `predicted_class`, `confidence`, and class probabilities:

```powershell
python -m driverisknet.infer --checkpoint artifacts/checkpoints/lstm_best.pt --dataset data/synthetic_v1_1.npz --index 0
python -m driverisknet.infer --checkpoint artifacts/checkpoints/mlp_best.pt --dataset data/synthetic_v1_1.npz --index 0
```

Use `--input sequence.json` for a 32 × 7 JSON array in the documented feature order.

## Tests and artifacts

```powershell
python -m pytest -q
```

The suite checks sequence shape, random split availability, zero group overlap, non-trivial single-feature separability, MLP/LSTM output shapes, both checkpoint types, and error-analysis output.

- `artifacts/{mlp,lstm}_history.json`
- `artifacts/{mlp,lstm}_learning_curve.png`
- `artifacts/{mlp,lstm}_error_analysis.json`
- `artifacts/error_analysis.json` (LSTM high-confidence errors)
- `artifacts/baselines.json`
- `artifacts/results_{mlp,lstm}.json`
- `artifacts/multiseed_results.{json,csv}`
- `artifacts/{mlp,lstm}_confusion_matrix.png`

## Limitations

- Synthetic data, despite added noise and overlap.
- No camera, lidar, map, or environment context.
- No real-driver, real-vehicle, or closed-course validation.
- Sequence classification only; no forecasting, localization, planning, or control.
- Not calibrated or validated for safety-critical production use.
- Quantile-based labels preserve the intended imbalance and are not a substitute for expert annotation.
