# DriveRiskNet V1.1 — Realism Upgrade

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

**This is synthetic data for training-pipeline demonstration and does not represent real-world autonomous-driving validation performance.**

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

Training fixes seed 42, applies training-set normalization, inverse-frequency class weights, AdamW, gradient clipping, validation-loss early stopping, and best-checkpoint restoration. JSON histories record train/validation loss and macro-F1; PNG learning curves are generated from those histories.

## Results

Actual CPU results from a clean V1.1 group-split run:

| Model | Params | Best / run epochs | Val accuracy | Val macro-F1 | Test accuracy | Test precision | Test recall | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 12,051 | 17 / 23 | 0.6689 | 0.6107 | 0.6500 | 0.5721 | 0.6284 | 0.5840 |
| LSTM | 11,187 | 15 / 21 | 0.7022 | 0.6551 | 0.6500 | 0.5722 | 0.6390 | 0.5833 |

MLP test confusion matrix (rows true, columns predicted; normal/warning/critical):

```text
[[400, 136, 27],
 [ 42, 110, 77],
 [  3,  30, 75]]
```

LSTM test confusion matrix:

```text
[[401, 137, 25],
 [ 35, 102, 92],
 [  0,  26, 82]]
```

The LSTM is stronger on validation, but its test macro-F1 is 0.0007 below the MLP, so this run does **not** establish a reliable LSTM advantage. Much of the synthetic risk signal can be recovered from static extremes and levels; temporal evolution helps some critical recall but also moves more borderline warnings into critical.

## Baselines

| Baseline | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Majority class | 0.6256 | 0.2566 |
| Single-feature minimum-TTC heuristic | 0.6178 | 0.4928 |

The neural models substantially improve macro-F1 over both sanity baselines. The leakage test also trains shallow classifiers on min/mean/max summaries from one feature at a time and rejects the dataset if any reaches 0.90 accuracy.

## Error Analysis

Both models most often confuse `warning` with its neighboring classes, consistent with deliberate distribution overlap. Among the 20 highest-confidence errors, brake-under-response and routine borderline episodes dominate. The LSTM produces more warning→critical errors, while recovering a slightly larger share of truly critical examples. Machine-readable summaries include true/predicted labels, confidence, case, and key signal statistics.

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

## Limitations

- Synthetic data, despite added noise and overlap.
- No camera, lidar, map, or environment context.
- No real-driver, real-vehicle, or closed-course validation.
- Sequence classification only; no forecasting, localization, planning, or control.
- Not calibrated or validated for safety-critical production use.
- Quantile-based labels preserve the intended imbalance and are not a substitute for expert annotation.
