from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import CLASS_NAMES, FEATURE_NAMES, DataConfig

CASE_NAMES = ["routine", "harsh_braking", "low_ttc_recovery", "lateral_instability",
              "ttc_collapse", "brake_under_response", "noisy_normal", "borderline"]


def _smooth(values, width=3):
    return np.convolve(values, np.ones(width) / width, mode="same")


def generate_synthetic_data(config: DataConfig):
    """Generate features from latent factors, then assign noisy composite-risk labels."""
    rng = np.random.default_rng(config.seed)
    n, length = config.num_samples, config.sequence_length
    num_groups = int(np.ceil(n / config.group_size))
    groups = np.repeat(np.arange(num_groups), config.group_size)[:n].astype(np.int64)
    x = np.empty((n, length, len(FEATURE_NAMES)), dtype=np.float32)
    risks, case_ids = np.empty(n, np.float32), np.empty(n, np.int64)
    case_probs = np.array([.36, .10, .10, .10, .09, .08, .09, .08])
    profiles = [{"speed": rng.normal(0, 4), "aggression": rng.normal(0, .35),
                 "bias": rng.normal(0, .10, 7), "noise": rng.uniform(.75, 1.35)}
                for _ in range(num_groups)]
    base_time = np.linspace(0, 1, length)

    for i, group in enumerate(groups):
        p = profiles[int(group)]
        case = int(rng.choice(len(CASE_NAMES), p=case_probs)); case_ids[i] = case
        collision = np.clip(rng.beta(1.5, 3.2) + .12 * p["aggression"], 0, 1)
        lateral = np.clip(rng.beta(1.4, 4.0) + .10 * p["aggression"], 0, 1)
        braking = np.clip(rng.beta(1.5, 3.0) + .12 * collision, 0, 1)
        recovery, under = rng.beta(2.2, 2.0), rng.beta(1.2, 5.0)
        noise = rng.uniform(.6, 1.4) * p["noise"]
        # Cases shift several continuous factors; no case determines a class.
        if case == 1:
            braking += rng.uniform(.35, .65); recovery += rng.uniform(.15, .35); collision -= .12
        elif case == 2:
            collision += rng.uniform(.25, .50); recovery += rng.uniform(.35, .60); braking += .15
        elif case == 3:
            lateral += rng.uniform(.35, .65); collision += .08
        elif case == 4:
            collision += rng.uniform(.45, .75); braking += .15; recovery -= .20
        elif case == 5:
            collision += rng.uniform(.30, .60); under += rng.uniform(.40, .70); braking -= .15
        elif case == 6:
            noise += rng.uniform(.8, 1.3); recovery += .15; collision -= .10
        elif case == 7:
            collision += rng.uniform(.20, .45); lateral += rng.uniform(.15, .35); recovery += rng.uniform(.05, .25)
        collision, lateral, braking, recovery, under = np.clip(
            [collision, lateral, braking, recovery, under], 0, 1)

        onset, duration = rng.uniform(.22, .72), rng.uniform(.18, .55)
        time = np.clip(base_time + _smooth(rng.normal(0, .014, length)), 0, 1)
        ramp = 1 / (1 + np.exp(-(time - onset) / max(.025, duration / 8)))
        recover_ramp = np.clip((time - min(.92, onset + duration)) / .18, 0, 1)
        danger = np.clip(ramp - recovery * .75 * recover_ramp, 0, 1)
        side = rng.choice([-1., 1.])
        speed0 = np.clip(rng.normal(21 + p["speed"], 6.5), 4, 38)
        long_acc = rng.normal(0, .35 * noise, length) - (.4 + 3 * braking) * danger
        lat_acc = rng.normal(0, .28 * noise, length) + side * (.25 + 1.9 * lateral) * danger
        yaw = rng.normal(0, .025 * noise, length) + side * (.025 + .18 * lateral) * danger
        wheel = rng.normal(0, .48 * noise, length) + side * (.3 + 2.1 * lateral) * danger
        delay = int(rng.integers(1, 7)); brake_signal = np.roll(danger, delay); brake_signal[:delay] = 0
        brake = rng.normal(3.5, 2.2 * noise, length) + 52 * braking * (1 - .7 * under) * brake_signal
        ttc = rng.uniform(6, 11) - (2 + 6.3 * collision) * danger + 2.8 * recovery * recover_ramp
        ttc += _smooth(rng.normal(0, .65 * noise, length))
        speed = speed0 + np.cumsum(long_acc) * (.1 / 3.6) + rng.normal(0, .35 * noise, length)
        values = np.stack([speed, long_acc, lat_acc, yaw, wheel, brake, ttc], axis=1) + p["bias"]
        values[:, 0] = np.clip(values[:, 0], 0, 45); values[:, 5] = np.clip(values[:, 5], 0, 100)
        values[:, 6] = np.clip(values[:, 6], .25, 14)
        # Sparse missing readings use last-value hold; some channels are quantized/clipped.
        for row, col in zip(*np.where(rng.random(values.shape) < .004)):
            values[row, col] = values[max(0, row - 1), col]
        if rng.random() < .08:
            col, step = int(rng.choice([1, 2, 4, 5, 6])), float(rng.choice([.1, .25, .5]))
            values[:, col] = np.round(values[:, col] / step) * step
        x[i] = values
        risks[i] = (1.15 * collision + .70 * lateral + .38 * braking + .90 * under
                    - .78 * recovery + collision * (.55 + under) + .55 * collision * lateral
                    + .10 * p["aggression"] + rng.normal(0, .42))

    low, high = np.quantile(risks, [config.normal_ratio, 1 - config.critical_ratio])
    labels = np.where(risks >= high, 2, np.where(risks >= low, 1, 0)).astype(np.int64)
    boundary = (np.abs(risks - low) < .16) | (np.abs(risks - high) < .16)
    flip = boundary & (rng.random(n) < .12)
    labels[flip] = np.clip(labels[flip] + rng.choice([-1, 1], flip.sum()), 0, 2)
    return x, labels, groups, case_ids


def save_dataset(path, x, y, groups, case_ids, config):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y, groups=groups, case_ids=case_ids,
                        feature_names=FEATURE_NAMES, class_names=CLASS_NAMES,
                        case_names=CASE_NAMES, config=asdict(config))


def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.int64), data["groups"], data["case_ids"]


def split_indices(y, groups, seed=42, mode="group"):
    indices = np.arange(len(y))
    if mode == "random":
        train, temp = train_test_split(indices, test_size=.30, random_state=seed, stratify=y)
        val, test = train_test_split(temp, test_size=.50, random_state=seed, stratify=y[temp])
    elif mode == "group":
        train, temp = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed).split(indices, y, groups))
        vr, tr = next(GroupShuffleSplit(n_splits=1, test_size=.50, random_state=seed + 1).split(temp, y[temp], groups[temp]))
        val, test = temp[vr], temp[tr]
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return train, val, test


def split_data(x, y, groups, seed=42, mode="group"):
    ids = split_indices(y, groups, seed, mode)
    return tuple((x[idx], y[idx]) for idx in ids), ids


class SequenceDataset(Dataset):
    def __init__(self, x, y=None, mean=None, std=None):
        self.mean = x.mean((0, 1), keepdims=True) if mean is None else mean
        self.std = x.std((0, 1), keepdims=True) + 1e-6 if std is None else std
        self.x = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx] if self.y is None else (self.x[idx], self.y[idx])


def make_loaders(splits, batch_size=64, seed=42):
    train = SequenceDataset(*splits[0]); val = SequenceDataset(*splits[1], mean=train.mean, std=train.std)
    test = SequenceDataset(*splits[2], mean=train.mean, std=train.std)
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
            DataLoader(val, batch_size=batch_size), DataLoader(test, batch_size=batch_size), train.mean, train.std)
