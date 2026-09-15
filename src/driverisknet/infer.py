import argparse
import json

import numpy as np
import torch

from .config import CLASS_NAMES
from .models import build_model


def load_predictor(checkpoint):
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(bundle["model_name"], **bundle["model_kwargs"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, np.asarray(bundle["mean"]), np.asarray(bundle["std"]), bundle


def predict(checkpoint, sequence):
    model, mean, std, bundle = load_predictor(checkpoint)
    sequence = np.asarray(sequence, dtype=np.float32)
    expected = (bundle["model_kwargs"]["sequence_length"], bundle["model_kwargs"]["num_features"])
    if sequence.shape != expected:
        raise ValueError(f"Expected input shape {expected}, got {sequence.shape}")
    x = torch.from_numpy(((sequence[None] - mean) / std).astype(np.float32))
    with torch.no_grad():
        probabilities = torch.softmax(model(x), 1)[0].numpy()
    index = int(probabilities.argmax())
    return {"class_id": index, "predicted_class": CLASS_NAMES[index], "confidence": float(probabilities[index]),
            "probabilities": dict(zip(CLASS_NAMES, probabilities.tolist()))}


def main():
    parser = argparse.ArgumentParser(description="Run DriveRiskNet inference for one JSON vehicle-signal sequence.")
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSON file containing a [time, features] array")
    source.add_argument("--dataset", help="NPZ dataset; use one sequence selected by --index")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.input:
        with open(args.input, encoding="utf-8") as handle:
            sequence = json.load(handle)
    else:
        sequence = np.load(args.dataset, allow_pickle=True)["x"][args.index]
    print(json.dumps(predict(args.checkpoint, sequence), indent=2))


if __name__ == "__main__": main()
