import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .plotting import save_confusion_matrix
from .train import run_training
from .utils import save_json

SEEDS = [42, 123, 2026]
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
RUN_FIELDS = {"seed", "model", "best_epoch", "train_time_sec", "params", "val_accuracy",
              "val_macro_precision", "val_macro_recall", "val_macro_f1", "test_accuracy",
              "test_macro_precision", "test_macro_recall", "test_macro_f1"}


def validate_results_schema(payload):
    if set(payload["protocol"]["seeds"]) != {42, 123, 2026}:
        raise ValueError("Required seeds are missing")
    if len(payload["runs"]) != 6 or any(not RUN_FIELDS <= set(run) for run in payload["runs"]):
        raise ValueError("Multi-seed run schema is incomplete")
    return True


def summarize_runs(runs):
    summary = {}
    for model in ("mlp", "lstm"):
        model_runs = [r for r in runs if r["model"] == model]
        values = {"val_macro_f1": [r["val_macro_f1"] for r in model_runs],
                  "test_accuracy": [r["test_accuracy"] for r in model_runs],
                  "test_macro_precision": [r["test_macro_precision"] for r in model_runs],
                  "test_macro_recall": [r["test_macro_recall"] for r in model_runs],
                  "test_macro_f1": [r["test_macro_f1"] for r in model_runs],
                  "train_time_sec": [r["train_time_sec"] for r in model_runs]}
        summary[model] = {key: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=0))}
                          for key, v in values.items()}
        f1 = values["test_macro_f1"]
        summary[model]["stability"] = {"min": float(min(f1)), "max": float(max(f1)),
                                         "range": float(max(f1) - min(f1)),
                                         "std": float(np.std(f1, ddof=0))}
    return summary


def select_representative_seed(runs, model):
    model_runs = [r for r in runs if r["model"] == model]
    mean = np.mean([r["test_macro_f1"] for r in model_runs])
    return min(model_runs, key=lambda r: (abs(r["test_macro_f1"] - mean), r["seed"]))["seed"]


def _flat_run(seed, model, result):
    row = {"seed": seed, "model": model, "best_epoch": result["best_epoch"],
           "train_time_sec": result["training_seconds"], "params": result["parameters"]}
    for split in ("validation", "test"):
        prefix = "val" if split == "validation" else "test"
        for metric in METRICS:
            row[f"{prefix}_{'macro_' if metric != 'accuracy' else ''}{metric.replace('_macro', '')}"] = result[split][metric]
    # Stable, explicit names used by schema and aggregation.
    row.update({"val_accuracy": result["validation"]["accuracy"],
                "val_macro_precision": result["validation"]["precision_macro"],
                "val_macro_recall": result["validation"]["recall_macro"],
                "val_macro_f1": result["validation"]["f1_macro"],
                "test_accuracy": result["test"]["accuracy"],
                "test_macro_precision": result["test"]["precision_macro"],
                "test_macro_recall": result["test"]["recall_macro"],
                "test_macro_f1": result["test"]["f1_macro"]})
    return row


def run_benchmark(output_dir="artifacts", seeds=SEEDS):
    started = time.perf_counter(); root = Path(output_dir); runs, full = [], {}
    for seed in seeds:
        for model in ("mlp", "lstm"):
            run_dir = root / "multiseed" / f"seed_{seed}" / model
            result = run_training(model, 6000, 35, run_dir, seed, "group", data_seed=42, split_seed=42)
            runs.append(_flat_run(seed, model, result)); full[(seed, model)] = (result, run_dir)
    summary = summarize_runs(runs)
    representatives = {model: select_representative_seed(runs, model) for model in ("mlp", "lstm")}
    for model, seed in representatives.items():
        result, run_dir = full[(seed, model)]
        save_confusion_matrix(result["test"]["confusion_matrix"], root / f"{model}_confusion_matrix.png",
                              f"{model.upper()} confusion matrix (seed {seed})")
        shutil.copy2(run_dir / f"{model}_history.json", root / f"{model}_history.json")
        shutil.copy2(run_dir / f"{model}_learning_curve.png", root / f"{model}_learning_curve.png")
        shutil.copy2(run_dir / f"{model}_error_analysis.json", root / f"{model}_error_analysis.json")
    lstm_dir = full[(representatives["lstm"], "lstm")][1]
    shutil.copy2(lstm_dir / "lstm_error_analysis.json", root / "error_analysis.json")
    elapsed = time.perf_counter() - started
    payload = {"protocol": {"seeds": list(seeds), "dataset_seed": 42, "split_seed": 42,
                             "split": "group", "std": "population", "num_samples": 6000,
                             "sequence_length": 32}, "runs": runs, "summary": summary,
               "representative_seed": representatives, "total_runtime_sec": elapsed}
    validate_results_schema(payload); save_json(payload, root / "multiseed_results.json")
    with (root / "multiseed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0])); writer.writeheader(); writer.writerows(runs)
    print(json.dumps(payload, indent=2)); return payload


def main():
    parser = argparse.ArgumentParser(description="Run the fixed DriveRiskNet multi-seed benchmark.")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args(); run_benchmark(args.output_dir)


if __name__ == "__main__": main()
