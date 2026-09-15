import numpy as np

from driverisknet.multiseed import RUN_FIELDS, select_representative_seed, summarize_runs, validate_results_schema


def _runs():
    rows = []
    for model in ("mlp", "lstm"):
        for seed, value in zip((42, 123, 2026), (.50, .60, .90)):
            row = {field: 1.0 for field in RUN_FIELDS}
            row.update({"model": model, "seed": seed, "best_epoch": 3, "params": 100,
                        "val_macro_f1": value - .01, "test_macro_f1": value,
                        "test_accuracy": value, "test_macro_precision": value,
                        "test_macro_recall": value, "train_time_sec": 2.0})
            rows.append(row)
    return rows


def test_multiseed_results_schema():
    payload = {"protocol": {"seeds": [42, 123, 2026]}, "runs": _runs()}
    assert validate_results_schema(payload)


def test_multiseed_summary_matches_raw_runs():
    runs = _runs(); summary = summarize_runs(runs)
    expected = np.mean([.50, .60, .90])
    assert np.isclose(summary["mlp"]["test_macro_f1"]["mean"], expected)
    assert np.isclose(summary["lstm"]["test_macro_f1"]["std"], np.std([.50, .60, .90]))


def test_representative_seed_is_not_best_seed_by_default():
    runs = _runs()
    assert select_representative_seed(runs, "mlp") == 123
    assert select_representative_seed(runs, "mlp") != 2026
