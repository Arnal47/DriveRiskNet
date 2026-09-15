from dataclasses import asdict, dataclass


FEATURE_NAMES = [
    "speed",
    "longitudinal_accel",
    "lateral_accel",
    "yaw_rate",
    "wheel_speed_diff",
    "brake_pressure",
    "ttc",
]
CLASS_NAMES = ["normal", "warning", "critical"]


@dataclass
class DataConfig:
    num_samples: int = 6000
    sequence_length: int = 32
    seed: int = 42
    group_size: int = 20
    normal_ratio: float = 0.65
    warning_ratio: float = 0.23
    critical_ratio: float = 0.12

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    model: str = "lstm"
    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    hidden_size: int = 48
    num_layers: int = 1
    dropout: float = 0.2
    seed: int = 42

    def to_dict(self):
        return asdict(self)
