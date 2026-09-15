import numpy as np

from .config import CLASS_NAMES, FEATURE_NAMES
from .data import CASE_NAMES


def build_error_analysis(x, y_true, probabilities, case_ids, limit=20):
    y_pred = probabilities.argmax(1); confidence = probabilities.max(1)
    errors = np.flatnonzero(y_pred != y_true)
    errors = errors[np.argsort(confidence[errors])[::-1]][:limit]
    rows = []
    for idx in errors:
        seq = x[idx]
        rows.append({"sample_index": int(idx), "true_label": CLASS_NAMES[int(y_true[idx])],
                     "predicted_label": CLASS_NAMES[int(y_pred[idx])], "confidence": float(confidence[idx]),
                     "case": CASE_NAMES[int(case_ids[idx])],
                     "key_feature_summary": {
                         "mean_speed": float(seq[:, 0].mean()), "min_longitudinal_accel": float(seq[:, 1].min()),
                         "max_abs_lateral_accel": float(np.abs(seq[:, 2]).max()),
                         "max_abs_yaw_rate": float(np.abs(seq[:, 3]).max()),
                         "max_wheel_speed_diff": float(np.abs(seq[:, 4]).max()),
                         "max_brake_pressure": float(seq[:, 5].max()), "min_ttc": float(seq[:, 6].min())}})
    return {"total_errors": int((y_pred != y_true).sum()), "saved_errors": len(rows), "errors": rows}

