import pytest
import torch

from driverisknet.models import build_model


@pytest.mark.parametrize("name", ["mlp", "lstm"], ids=["mlp_forward_shape", "lstm_forward_shape"])
def test_model_forward_shape(name):
    model = build_model(name, sequence_length=32, num_features=7)
    assert model(torch.randn(5, 32, 7)).shape == (5, 3)
