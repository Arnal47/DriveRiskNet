import numpy as np
from driverisknet.error_analysis import build_error_analysis


def test_error_analysis_output():
    x = np.zeros((3, 8, 7), dtype=np.float32)
    y = np.array([0, 1, 2]); probs = np.array([[.1, .8, .1], [.1, .8, .1], [.7, .2, .1]])
    result = build_error_analysis(x, y, probs, np.array([0, 7, 5]))
    assert result["total_errors"] == 2 and result["saved_errors"] == 2
    assert {"true_label", "predicted_label", "confidence", "key_feature_summary"} <= result["errors"][0].keys()
