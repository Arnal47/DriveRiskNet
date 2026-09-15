import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from .baselines import fit_ttc_heuristic, majority_baseline, ttc_heuristic
from .config import DataConfig, FEATURE_NAMES, TrainConfig
from .data import generate_synthetic_data, make_loaders, save_dataset, split_data
from .error_analysis import build_error_analysis
from .models import build_model, count_parameters
from .plotting import save_learning_curve
from .training import evaluate, fit, predict_loader
from .utils import save_json, set_seed


def run_training(model_name="lstm", num_samples=6000, epochs=35, output_dir="artifacts", seed=42,
                 split_mode="group", data_seed=42, split_seed=42):
    started = time.perf_counter(); set_seed(seed); output_dir = Path(output_dir)
    data_config = DataConfig(num_samples=num_samples, seed=data_seed)
    train_config = TrainConfig(model=model_name, epochs=epochs, seed=seed)
    x, y, groups, case_ids = generate_synthetic_data(data_config)
    save_dataset(output_dir.parent / "data" / "synthetic_v1_1.npz", x, y, groups, case_ids, data_config)
    splits, indices = split_data(x, y, groups, split_seed, split_mode)
    train_loader, val_loader, test_loader, mean, std = make_loaders(splits, train_config.batch_size, seed)
    train_y = splits[0][1]
    weights = compute_class_weight("balanced", classes=np.arange(3), y=train_y)
    model = build_model(model_name, data_config.sequence_length, len(FEATURE_NAMES), train_config.hidden_size,
                        train_config.num_layers, train_config.dropout)
    checkpoint = output_dir / "checkpoints" / f"{model_name}_best.pt"
    history = fit(model, train_loader, val_loader, weights, train_config, checkpoint)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    val_metrics, test_metrics = evaluate(model, val_loader, loss_fn, "cpu"), evaluate(model, test_loader, loss_fn, "cpu")
    probabilities, test_targets = predict_loader(model, test_loader)
    test_idx = indices[2]
    errors = build_error_analysis(x[test_idx], test_targets, probabilities, case_ids[test_idx])
    thresholds = fit_ttc_heuristic(x[indices[0]], y[indices[0]])
    baselines = {"majority": majority_baseline(y[indices[0]], y[test_idx]),
                 "ttc_heuristic": ttc_heuristic(x[test_idx], y[test_idx], thresholds)}
    bundle = {"model_name": model_name,
              "model_kwargs": {"sequence_length": data_config.sequence_length, "num_features": len(FEATURE_NAMES),
                               "hidden_size": train_config.hidden_size, "num_layers": train_config.num_layers,
                               "dropout": train_config.dropout},
              "state_dict": model.state_dict(), "mean": mean, "std": std, "feature_names": FEATURE_NAMES}
    torch.save(bundle, checkpoint)
    save_json({"history": history}, output_dir / f"{model_name}_history.json")
    save_learning_curve(history, output_dir / f"{model_name}_learning_curve.png")
    save_json(errors, output_dir / f"{model_name}_error_analysis.json")
    if model_name == "lstm": save_json(errors, output_dir / "error_analysis.json")
    save_json(baselines, output_dir / "baselines.json")
    elapsed = time.perf_counter() - started
    result = {"version": "1.1", "data_config": data_config.to_dict(), "train_config": train_config.to_dict(),
              "split_mode": split_mode, "data_seed": data_seed, "split_seed": split_seed,
              "split_sizes": dict(zip(["train", "validation", "test"], [len(i) for i in indices])),
              "split_group_counts": dict(zip(["train", "validation", "test"], [len(np.unique(groups[i])) for i in indices])),
              "class_counts": {str(i): int((y == i).sum()) for i in range(3)},
              "class_weights": weights.tolist(), "parameters": count_parameters(model),
              "epochs_completed": len(history), "best_epoch": int(np.argmin([h["val_loss"] for h in history]) + 1),
              "validation": val_metrics, "test": test_metrics, "baselines": baselines,
              "error_analysis": {k: v for k, v in errors.items() if k != "errors"},
              "training_seconds": elapsed}
    save_json(result, output_dir / f"results_{model_name}.json")
    print(result); return result


def main():
    parser = argparse.ArgumentParser(description="Train DriveRiskNet V1.1.")
    parser.add_argument("--model", choices=["mlp", "lstm"], default="lstm")
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["group", "random"], default="group")
    args = parser.parse_args()
    run_training(args.model, args.num_samples, args.epochs, args.output_dir, args.seed, args.split)


if __name__ == "__main__": main()
