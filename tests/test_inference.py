import numpy as np
import torch

from driverisknet.infer import predict
from driverisknet.models import build_model


def test_checkpoint_inference_smoke(tmp_path):
    model = build_model("lstm", sequence_length=8, num_features=7, hidden_size=12)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_name": "lstm", "model_kwargs": {"sequence_length": 8, "num_features": 7,
               "hidden_size": 12, "num_layers": 1, "dropout": 0.2}, "state_dict": model.state_dict(),
               "mean": np.zeros((1, 1, 7)), "std": np.ones((1, 1, 7))}, checkpoint)
    result = predict(checkpoint, np.zeros((8, 7), dtype=np.float32))
    assert result["predicted_class"] in {"normal", "warning", "critical"}
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_mlp_checkpoint_inference_smoke(tmp_path):
    model = build_model("mlp", sequence_length=8, num_features=7, hidden_size=12)
    checkpoint = tmp_path / "mlp.pt"
    torch.save({"model_name": "mlp", "model_kwargs": {"sequence_length": 8, "num_features": 7,
               "hidden_size": 12, "num_layers": 1, "dropout": 0.2}, "state_dict": model.state_dict(),
               "mean": np.zeros((1, 1, 7)), "std": np.ones((1, 1, 7))}, checkpoint)
    assert predict(checkpoint, np.zeros((8, 7), np.float32))["predicted_class"]
