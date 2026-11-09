"""Training-only per-axis univariate filtering of motion and light features."""
import numpy as np
import common


def get_features(data):
    return np.nan_to_num(np.concatenate([data['M'], data['F']], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


CONFIGS = [
    ('exp13_select64_svm', dict(family='svm', select_k=64, C=10, gamma='scale')),
    ('exp13_select128_trees', dict(family='trees', select_k=128,
                                 n_estimators=220, min_samples_leaf=2,
                                 max_features=.6)),
]


if __name__ == '__main__':
    features = get_features(common.load())
    for name, config in CONFIGS:
        common.evaluate(name, features, config)
