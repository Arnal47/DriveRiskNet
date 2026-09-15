import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from driverisknet.config import DataConfig
from driverisknet.data import generate_synthetic_data, split_indices


def _sample(n=600):
    return generate_synthetic_data(DataConfig(num_samples=n, sequence_length=32, group_size=10, seed=7))


def test_dataset_sequence_shape():
    x, y, groups, cases = _sample(300)
    assert x.shape == (300, 32, 7) and y.shape == groups.shape == cases.shape == (300,)
    assert np.isfinite(x).all() and set(y) == {0, 1, 2}


def test_group_split_has_no_overlap():
    _, y, groups, _ = _sample()
    train, val, test = split_indices(y, groups, seed=7, mode="group")
    group_sets = [set(groups[idx]) for idx in (train, val, test)]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])


def test_random_split_is_available():
    _, y, groups, _ = _sample(300)
    splits = split_indices(y, groups, seed=7, mode="random")
    assert sorted(np.concatenate(splits).tolist()) == list(range(300))


def test_no_trivial_label_leakage():
    x, y, _, _ = _sample(1200)
    train, _, test = split_indices(y, np.arange(len(y)), seed=11, mode="random")
    scores = []
    for feature in range(x.shape[2]):
        summary = np.stack([x[:, :, feature].min(1), x[:, :, feature].mean(1), x[:, :, feature].max(1)], 1)
        tree = DecisionTreeClassifier(max_depth=2, random_state=3).fit(summary[train], y[train])
        scores.append(accuracy_score(y[test], tree.predict(summary[test])))
    assert max(scores) < .90, f"single-feature score suggests leakage: {scores}"
